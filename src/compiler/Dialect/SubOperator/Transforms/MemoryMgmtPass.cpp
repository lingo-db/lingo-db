#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/helper.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Debug.h"

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

   bool definesColumn(mlir::Attribute attr, tuples::Column* col) {
      if (!attr) return false;
      if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
         for (auto x : arrayAttr) {
            if (definesColumn(x, col)) return true;
         }
      } else if (auto mappingDefAttr = mlir::dyn_cast<subop::ColumnDefMemberMappingAttr>(attr)) {
         for (auto x : mappingDefAttr.getMapping()) {
            if (definesColumn(x.second, col)) return true;
         }
      } else if (auto columnDefAttr = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
         if (&columnDefAttr.getColumn() == col) return true;
      }
      return false;
   }

   mlir::Operation* findCreatorInStream(mlir::Value stream, tuples::Column* col) {
      while (stream) {
         mlir::Operation* defOp = stream.getDefiningOp();
         if (!defOp) return nullptr;
         for (auto attr : defOp->getAttrs()) {
            if (definesColumn(attr.getValue(), col)) {
               return defOp;
            }
         }
         if (defOp->getNumOperands() > 0 && mlir::isa<tuples::TupleStreamType>(defOp->getOperand(0).getType())) {
            stream = defOp->getOperand(0);
         } else {
            break;
         }
      }
      return nullptr;
   }

   bool needsPromotion(tuples::Column* col, mlir::Value stream) {
      if (!typeNeedsManagement(col->type)) return false;
      auto* creator = findCreatorInStream(stream, col);
      if (creator) {
         if (auto gatherOp = mlir::dyn_cast<subop::GatherOp>(creator)) {
            if (!mlir::isa<subop::TableEntryRefType>(gatherOp.getRef().getColumn().type)) {
               return false;
            }
            return true;
         }
      }
      return false;
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
         } else {
            assert(!typeNeedsManagement(listType.getElementType()));
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
            auto managed = mlir::dyn_cast<db::ManagedType>(operand.get().getType());
            if (!managed || !managed.needsManagement()) continue;
            if (notCounted.contains(operand.get())) continue;
            mlir::OpBuilder builder(block->getTerminator());
            mlir::Value newVal = managed.emitPromoteToGlobal(builder, block->getTerminator()->getLoc(), operand.get());
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

      // Data before materialize should be promoted to a global lifetime.
      // When storing data persistently in a materialized table,
      // we must clone/promote them into a longer-living global representation.
      module.walk([&](subop::MaterializeOp materializeOp) {
         // Step 1: Identify which columns being materialized need to be promoted and create new column definitions.
         std::vector<std::pair<tuples::ColumnRefAttr, tuples::ColumnDefAttr>> promoteCols;
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> newMapping;

         for (auto pair : materializeOp.getMapping().getMapping()) {
            if (needsPromotion(&pair.second.getColumn(), materializeOp.getStream())) {
               auto& columnManager = pair.second.getColumn().type.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
               std::string scopeName = columnManager.getUniqueScope("promote");
               std::string attributeName = pair.second.getName().getLeafReference().getValue().str();
               tuples::ColumnDefAttr newDef = columnManager.createDef(scopeName, attributeName);
               newDef.getColumn().type = pair.second.getColumn().type;
               tuples::ColumnRefAttr newRef = columnManager.createRef(scopeName, attributeName);
               promoteCols.push_back({pair.second, newDef});
               newMapping.push_back({pair.first, newRef});
            } else {
               newMapping.push_back(pair);
            }
         }

         // Step 2: If any columns need promotion, insert a MapOp before the MaterializeOp to perform the promotion.
         if (!promoteCols.empty()) {
            mlir::OpBuilder builder(materializeOp);

            std::vector<mlir::Attribute> computedCols;
            std::vector<mlir::Attribute> inputCols;
            for (auto& pair : promoteCols) {
               computedCols.push_back(pair.second);
               inputCols.push_back(pair.first);
            }

            auto mapOp = builder.create<subop::MapOp>(
               materializeOp.getLoc(),
               materializeOp.getStream(),
               builder.getArrayAttr(computedCols),
               builder.getArrayAttr(inputCols));

            mlir::Block* block = new mlir::Block();
            mapOp.getFn().push_back(block);

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(block);

            std::vector<mlir::Value> promotedValues;
            // Step 3: Populate the MapOp's body with the actual promotion logic
            for (auto& pair : promoteCols) {
               auto oldRef = pair.first;
               auto type = oldRef.getColumn().type;

               mlir::Value arg = block->addArgument(type, materializeOp.getLoc());

               auto managed = mlir::cast<db::ManagedType>(type);
               mlir::Value promoted = managed.emitPromoteToGlobal(builder, materializeOp.getLoc(), arg);
               promotedValues.push_back(promoted);
            }
            builder.create<tuples::ReturnOp>(materializeOp.getLoc(), promotedValues);

            // Step 4: Update the MaterializeOp to consume the stream from the MapOp and use the newly promoted columns.
            materializeOp->setOperand(0, mapOp.getResult());
            materializeOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(builder.getContext(), newMapping));
         }
      });

      // When inserting data into a hash table (LookupOrInsert), the keys might need to be promoted.
      module.walk([&](subop::LookupOrInsertOp lookupOp) {
         // Step 1: Identify which keys need to be promoted and create new column definitions for them.
         std::vector<std::pair<tuples::ColumnRefAttr, tuples::ColumnDefAttr>> promoteCols;
         llvm::SmallVector<mlir::Attribute> newKeys;

         for (auto keyAttr : lookupOp.getKeys()) {
            auto keyRef = mlir::cast<tuples::ColumnRefAttr>(keyAttr);
            if (needsPromotion(&keyRef.getColumn(), lookupOp.getStream())) {
               auto& columnManager = keyRef.getColumn().type.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
               std::string scopeName = columnManager.getUniqueScope("promote");
               std::string attributeName = keyRef.getName().getLeafReference().getValue().str();
               tuples::ColumnDefAttr newDef = columnManager.createDef(scopeName, attributeName);
               newDef.getColumn().type = keyRef.getColumn().type;
               tuples::ColumnRefAttr newRef = columnManager.createRef(scopeName, attributeName);
               promoteCols.push_back({keyRef, newDef});
               newKeys.push_back(newRef);
            } else {
               newKeys.push_back(keyAttr);
            }
         }

         // Step 2: If any keys need promotion, insert a MapOp before the LookupOrInsertOp to perform the promotion.
         if (!promoteCols.empty()) {
            mlir::OpBuilder builder(lookupOp);

            std::vector<mlir::Attribute> computedCols;
            std::vector<mlir::Attribute> inputCols;
            for (auto& pair : promoteCols) {
               computedCols.push_back(pair.second);
               inputCols.push_back(pair.first);
            }

            auto mapOp = builder.create<subop::MapOp>(
               lookupOp.getLoc(),
               lookupOp.getStream(),
               builder.getArrayAttr(computedCols),
               builder.getArrayAttr(inputCols));

            mlir::Block* block = new mlir::Block();
            mapOp.getFn().push_back(block);

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(block);

            // Step 3: Populate the MapOp's body with the actual promotion logic.
            std::vector<mlir::Value> promotedValues;
            for (auto& pair : promoteCols) {
               auto oldRef = pair.first;
               auto type = oldRef.getColumn().type;

               mlir::Value arg = block->addArgument(type, lookupOp.getLoc());

               auto managed = mlir::cast<db::ManagedType>(type);
               mlir::Value promoted = managed.emitPromoteToGlobal(builder, lookupOp.getLoc(), arg);
               promotedValues.push_back(promoted);
            }
            builder.create<tuples::ReturnOp>(lookupOp.getLoc(), promotedValues);

            // Step 4: Update the LookupOrInsertOp to consume the stream from the MapOp and use the newly promoted keys.
            lookupOp->setOperand(0, mapOp.getResult());
            lookupOp.setKeysAttr(builder.getArrayAttr(newKeys));
         }
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::subop::createMemoryMgmtPass() {
   return std::make_unique<MemoryMgmtPass>();
}
