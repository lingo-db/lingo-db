#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/helper.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

namespace {
using namespace lingodb::compiler::dialect;

class MemoryMgmtPass : public mlir::PassWrapper<MemoryMgmtPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryMgmtPass)
   llvm::StringRef getArgument() const override { return "subop-memory-mgmt"; }
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::scf::SCFDialect>();
      registry.insert<mlir::func::FuncDialect>();
      registry.insert<mlir::arith::ArithDialect>();
   }

   bool typeNeedsManagement(mlir::Type t) {
      if (auto managed = mlir::dyn_cast<db::ManagedType>(t)) {
         return managed.needsManagement();
      }
      return false;
   }

   mlir::Value promoteToGlobal(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, llvm::DenseSet<mlir::Value>& notCounted) {
      if (notCounted.contains(value)) return value;
      auto type = value.getType();
      auto managed = mlir::dyn_cast<db::ManagedType>(type);

      if (!managed || !managed.needsManagement()) return value;

      if (auto listType = mlir::dyn_cast<db::ListType>(type)) {
         mlir::Value promotedList = managed.emitPromoteToGlobal(builder, loc, value);
         if (typeNeedsManagement(listType.getElementType())) {
            auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto len = builder.create<db::ListLengthOp>(loc, promotedList);
            auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
            builder.create<mlir::scf::ForOp>(loc, zero, len, step, std::nullopt, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange) {
               mlir::Value element = b.create<db::ListGetOp>(loc, listType.getElementType(), promotedList, idx);
               mlir::Value promotedElement = promoteToGlobal(b, loc, element, notCounted);
               b.create<db::ListSetOp>(loc, promotedList, idx, promotedElement);
               b.create<mlir::scf::YieldOp>(loc);
            });
         }
         return promotedList;
      }

      return managed.emitPromoteToGlobal(builder, loc, value);
   }

   void addUse(mlir::Value val, mlir::Operation* insertBeforeOp, llvm::DenseSet<mlir::Value>& notCounted) {
      if (notCounted.contains(val)) return;
      if (!typeNeedsManagement(val.getType())) return;
      mlir::OpBuilder builder(insertBeforeOp);
      if (mlir::isa<mlir::TupleType>(val.getType())) {
         llvm::SmallVector<mlir::Value> unpacked;
         builder.createOrFold<util::UnPackOp>(unpacked, insertBeforeOp->getLoc(), val);
         for (auto element : unpacked) {
            addUse(element, insertBeforeOp, notCounted);
         }
      } else {
         mlir::cast<db::ManagedType>(val.getType()).emitAddUse(builder, insertBeforeOp->getLoc(), val);
      }
   }

   void addUseAfter(mlir::Value val, mlir::Operation* insertAfterOp, llvm::DenseSet<mlir::Value>& notCounted) {
      if (notCounted.contains(val)) return;
      if (!typeNeedsManagement(val.getType())) return;
      mlir::OpBuilder builder(insertAfterOp->getContext());
      builder.setInsertionPointAfter(insertAfterOp);
      if (mlir::isa<mlir::TupleType>(val.getType())) {
         llvm::SmallVector<mlir::Value> unpacked;
         builder.createOrFold<util::UnPackOp>(unpacked, insertAfterOp->getLoc(), val);
         auto* insertionPoint = &*builder.getInsertionPoint();
         for (auto element : unpacked) {
            addUseAfter(element, insertionPoint, notCounted);
         }
      } else {
         mlir::cast<db::ManagedType>(val.getType()).emitAddUse(builder, insertAfterOp->getLoc(), val);
      }
   }

   void cleanupUse(mlir::Operation* insertBeforeOp, mlir::Value val, llvm::DenseSet<mlir::Value>& notCounted) {
      if (notCounted.contains(val)) return;
      if (auto tupleType = mlir::dyn_cast<mlir::TupleType>(val.getType())) {
         mlir::OpBuilder builder(insertBeforeOp);
         llvm::SmallVector<mlir::Value> unpacked;
         builder.createOrFold<util::UnPackOp>(unpacked, insertBeforeOp->getLoc(), val);
         for (auto element : unpacked) {
            if (typeNeedsManagement(element.getType())) {
               cleanupUse(insertBeforeOp, element, notCounted);
            }
         }
         return;
      }
      if (!typeNeedsManagement(val.getType())) return;
      mlir::OpBuilder builder(insertBeforeOp);
      mlir::SymbolRefAttr elementFn;
      if (auto listType = mlir::dyn_cast<db::ListType>(val.getType())) {
         if (mlir::isa<db::StringType>(listType.getElementType())) {
            // Lists of strings need a per-element cleanup. Emit a shared helper
            // _cleanup_list_str(list) once per module and refer to it by symbol.
            auto loc = insertBeforeOp->getLoc();
            std::string name = "_cleanup_list_str";
            auto moduleOp = insertBeforeOp->getParentOfType<mlir::ModuleOp>();
            mlir::func::FuncOp cleanupFn = moduleOp.lookupSymbol<mlir::func::FuncOp>(name);
            if (!cleanupFn) {
               auto fnType = builder.getFunctionType({listType}, {});
               mlir::OpBuilder::InsertionGuard guard(builder);
               builder.setInsertionPointToStart(moduleOp.getBody());
               cleanupFn = builder.create<mlir::func::FuncOp>(loc, name, fnType);
               builder.setInsertionPointToStart(cleanupFn.addEntryBlock());
               mlir::Value list = cleanupFn.getArgument(0);
               auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
               auto len = builder.create<db::ListLengthOp>(loc, list);
               auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
               builder.create<mlir::scf::ForOp>(loc, zero, len, step, std::nullopt, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange) {
                  mlir::Value element = b.create<db::ListGetOp>(loc, listType.getElementType(), list, idx);
                  b.create<db::MemoryCleanupUse>(loc, element, mlir::SymbolRefAttr());
                  b.create<mlir::scf::YieldOp>(loc);
               });
               builder.create<mlir::func::ReturnOp>(loc);
            }
            elementFn = mlir::SymbolRefAttr::get(builder.getContext(), name);
         } else if (mlir::isa<db::ListType>(listType.getElementType())) {
            // Nested lists clean up their direct elements, but the element is
            // itself a managed list value, so the inner cleanup uses the plain
            // list cleanup path rather than recursing into leaf elements.
            auto loc = insertBeforeOp->getLoc();
            std::string name = "_cleanup_list_list";
            auto moduleOp = insertBeforeOp->getParentOfType<mlir::ModuleOp>();
            mlir::func::FuncOp cleanupFn = moduleOp.lookupSymbol<mlir::func::FuncOp>(name);
            if (!cleanupFn) {
               auto fnType = builder.getFunctionType({listType}, {});
               mlir::OpBuilder::InsertionGuard guard(builder);
               builder.setInsertionPointToStart(moduleOp.getBody());
               cleanupFn = builder.create<mlir::func::FuncOp>(loc, name, fnType);
               builder.setInsertionPointToStart(cleanupFn.addEntryBlock());
               mlir::Value list = cleanupFn.getArgument(0);
               auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
               auto len = builder.create<db::ListLengthOp>(loc, list);
               auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
               builder.create<mlir::scf::ForOp>(loc, zero, len, step, std::nullopt, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange) {
                  mlir::Value element = b.create<db::ListGetOp>(loc, listType.getElementType(), list, idx);
                  b.create<db::MemoryCleanupUse>(loc, element, mlir::SymbolRefAttr());
                  b.create<mlir::scf::YieldOp>(loc);
               });
               builder.create<mlir::func::ReturnOp>(loc);
            }
            elementFn = mlir::SymbolRefAttr::get(builder.getContext(), name);
         } else {
            assert(!typeNeedsManagement(listType.getElementType()));
         }
      } else if (auto structType = mlir::dyn_cast<db::StructType>(val.getType())) {
         bool needsFieldMgmt = false;
         for (auto fieldType : structType.getTypes()) {
            if (typeNeedsManagement(fieldType)) {
               needsFieldMgmt = true;
               break;
            }
         }
         if (needsFieldMgmt) {
            auto loc = insertBeforeOp->getLoc();
            std::string name = "_cleanup_struct_" + std::to_string(reinterpret_cast<uintptr_t>(structType.getAsOpaquePointer()));
            auto moduleOp = insertBeforeOp->getParentOfType<mlir::ModuleOp>();
            mlir::func::FuncOp cleanupFn = moduleOp.lookupSymbol<mlir::func::FuncOp>(name);
            if (!cleanupFn) {
               auto fnType = builder.getFunctionType({structType}, {});
               mlir::OpBuilder::InsertionGuard guard(builder);
               builder.setInsertionPointToStart(moduleOp.getBody());
               cleanupFn = builder.create<mlir::func::FuncOp>(loc, name, fnType);
               builder.setInsertionPointToStart(cleanupFn.addEntryBlock());
               mlir::Value strct = cleanupFn.getArgument(0);
               for (size_t i = 0; i < structType.getTypes().size(); ++i) {
                  auto fieldType = structType.getTypes()[i];
                  if (typeNeedsManagement(fieldType)) {
                     auto fieldNameAttr = mlir::cast<mlir::StringAttr>(structType.getNames()[i]);
                     mlir::Value element = builder.create<db::StructGetOp>(loc, fieldType, strct, fieldNameAttr);
                     builder.create<db::MemoryCleanupUse>(loc, element, mlir::SymbolRefAttr());
                  }
               }
               builder.create<mlir::func::ReturnOp>(loc);
            }
            elementFn = mlir::SymbolRefAttr::get(builder.getContext(), name);
         }
      }
      mlir::cast<db::ManagedType>(val.getType()).emitCleanupUse(builder, insertBeforeOp->getLoc(), val, elementFn);
   }

   void handleBlock(mlir::Block* block, mlir::Block* mapBlock, llvm::DenseSet<mlir::Value>& notCounted) {
      llvm::DenseSet<mlir::Value> returnedValues;
      auto* terminator = block->getTerminator();
      for (auto retVal : terminator->getOperands()) {
         returnedValues.insert(retVal);
      }
      std::vector<mlir::Value> valuesToManage;
      llvm::SmallVector<mlir::Operation*> ops;
      for (auto& op : block->getOperations()) {
         ops.push_back(&op);
      }
      for (auto* op : ops) {
         if (auto refCounted = mlir::dyn_cast<db::RefCountedOp>(op)) {
            mlir::OpBuilder builder(op);
            if (auto* rewritten = refCounted.rewriteForRefCount(builder, returnedValues)) {
               op = rewritten;
            } else {
               llvm::SmallVector<mlir::Value> owned;
               refCounted.getOwnedOperands(owned);
               for (auto v : owned) addUse(v, op, notCounted);
               llvm::SmallVector<mlir::Value> borrowed;
               refCounted.getBorrowedResults(borrowed);
               for (auto v : borrowed) addUseAfter(v, op, notCounted);
            }
         }
         for (auto result : op->getResults()) {
            if (typeNeedsManagement(result.getType())) {
               if (returnedValues.contains(result)) {
                  // produced and returned by this block — no extra bookkeeping needed
                  returnedValues.erase(result);
               } else {
                  valuesToManage.push_back(result);
               }
            }
         }
      }
      for (auto blockArg : block->getArguments()) {
         if (typeNeedsManagement(blockArg.getType())) {
            valuesToManage.push_back(blockArg);
         }
      }

      for (auto value : valuesToManage) {
         cleanupUse(block->getTerminator(), value, notCounted);
      }
      if (block == mapBlock) {
         // values returned from the subop.map fn outlive the per-row scope;
         // promote them to a global lifetime instead of bumping the refcount.
         for (auto& operand : terminator->getOpOperands()) {
            mlir::OpBuilder builder(block->getTerminator());
            mlir::Value newVal = promoteToGlobal(builder, block->getTerminator()->getLoc(), operand.get(), notCounted);
            operand.set(newVal);
         }
      } else {
         for (auto value : returnedValues) {
            if (typeNeedsManagement(value.getType())) {
               addUse(value, block->getTerminator(), notCounted);
            }
         }
      }
   }

   // Constants and values derived from them (cast/AsNullable/NullableGet) don't
   // own refcounts — track them so we skip emitting add_use / cleanup_use.
   void seedNotCounted(mlir::Region& region, llvm::DenseSet<mlir::Value>& notCounted) {
      region.walk([&](mlir::Operation* op) {
         if (mlir::isa<db::ConstantOp, db::NullOp>(op)) {
            notCounted.insert(op->getResult(0));
            return;
         }
         // py_interp.create_module returns a cached module owned by the
         // interpreter — do NOT decref it.
         // TODO: replace this hardcoded check with a proper "cached result"
         // marker on the op (e.g. a NotCounted trait or a dedicated entry on
         // RefCountedOpInterface), so the pass doesn't need to know about
         // specific ops.
         if (mlir::isa<py_interp::CreateModule>(op)) {
            notCounted.insert(op->getResult(0));
            return;
         }
         if (auto asNullableOp = mlir::dyn_cast<db::AsNullableOp>(op)) {
            if (notCounted.contains(asNullableOp.getVal())) {
               notCounted.insert(asNullableOp.getResult());
            }
         } else if (auto nullableGetOp = mlir::dyn_cast<db::NullableGetVal>(op)) {
            if (notCounted.contains(nullableGetOp.getVal())) {
               notCounted.insert(nullableGetOp.getResult());
            }
         } else if (auto castOp = mlir::dyn_cast<db::CastOp>(op)) {
            if (notCounted.contains(castOp.getVal()) &&
                mlir::isa<db::CharType>(getBaseType(castOp.getVal().getType())) &&
                mlir::isa<db::StringType>(getBaseType(castOp.getType()))) {
               notCounted.insert(castOp.getResult());
            }
         }
      });
   }

   void runOnOperation() override {
      auto module = getOperation();
      module.walk([&](subop::MapOp mapOp) {
         llvm::DenseSet<mlir::Value> notCounted;
         for (auto arg : mapOp.getFn().front().getArguments()) {
            if (mlir::isa<db::ListType>(arg.getType())) {
               continue;
            }
            notCounted.insert(arg);
         }
         seedNotCounted(mapOp.getFn(), notCounted);
         mapOp->walk([&](mlir::Block* block) {
            handleBlock(block, &mapOp.getFn().front(), notCounted);
         });
      });
      module.walk([&](mlir::func::FuncOp funcOp) {
         if (funcOp.isDeclaration()) return;
         if (funcOp.getName() == "main") return;
         llvm::DenseSet<mlir::Value> notCounted;
         seedNotCounted(funcOp.getBody(), notCounted);
         funcOp->walk([&](mlir::Block* block) {
            handleBlock(block, nullptr, notCounted);
         });
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::subop::createMemoryMgmtPass() {
   return std::make_unique<MemoryMgmtPass>();
}
