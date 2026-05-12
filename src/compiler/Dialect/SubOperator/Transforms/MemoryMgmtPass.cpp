#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/helper.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include <llvm/ADT/TypeSwitch.h>

namespace {
using namespace lingodb::compiler::dialect;

class MemoryMgmtPass : public mlir::PassWrapper<MemoryMgmtPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryMgmtPass)
   llvm::StringRef getArgument() const override { return "subop-memory-mgmt"; }

   bool typeNeedsManagement(mlir::Type t) {
      if (auto nullableType = mlir::dyn_cast<db::NullableType>(t)) {
         t = nullableType.getType();
      }
      if (mlir::isa<db::StringType, db::ListType>(t)) {
         return true;
      }
      // db.char<N> with N>1 lowers to the same refcounted VarLen32 as db.string.
      // db.char<1> is inline (i32) and doesn't need management.
      if (auto charType = mlir::dyn_cast<db::CharType>(t)) {
         return charType.getLen() > 1;
      }
      if (auto tupleType = mlir::dyn_cast<mlir::TupleType>(t)) {
         for (auto elementType : tupleType.getTypes()) {
            if (typeNeedsManagement(elementType)) {
               return true;
            }
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
         builder.create<db::MemoryAddUse>(insertBeforeOp->getLoc(), val);
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
         builder.create<db::MemoryAddUse>(insertAfterOp->getLoc(), val);
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
            builder.create<db::MemoryCleanupUse>(insertBeforeOp->getLoc(), val, mlir::SymbolRefAttr::get(builder.getContext(), name));
            return;
         }
         assert(!typeNeedsManagement(listType.getElementType()));
      }
      builder.create<db::MemoryCleanupUse>(insertBeforeOp->getLoc(), val, mlir::SymbolRefAttr());
   }

   void handleBlock(mlir::Block* block, mlir::Block* mapBlock, llvm::DenseSet<mlir::Value>& notCounted) {
      llvm::DenseSet<mlir::Value> returnedValues;
      auto terminator = block->getTerminator();
      for (auto retVal : terminator->getOperands()) {
         returnedValues.insert(retVal);
      }
      std::vector<mlir::Value> valuesToManage;
      llvm::SmallVector<mlir::Operation*> ops;
      for (auto& op : block->getOperations()) {
         ops.push_back(&op);
      }
      for (auto op : ops) {
         llvm::TypeSwitch<mlir::Operation*, void>(op)
            .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
               for (auto arg : forOp.getInitArgs()) {
                  addUse(arg, op, notCounted);
               }
            })
            .Case<mlir::scf::WhileOp>([&](mlir::scf::WhileOp whileOp) {
               for (auto arg : whileOp.getInits()) {
                  addUse(arg, op, notCounted);
               }
            })
            .Case<db::ListAppendOp>([&](db::ListAppendOp appendOp) {
               addUse(appendOp.getElement(), op, notCounted);
            })
            .Case<db::ListGetOp>([&](db::ListGetOp listGetOp) {
               addUseAfter(listGetOp.getElement(), op, notCounted);
            })
            .Case<util::UnPackOp>([&](util::UnPackOp unpackOp) {
               for (auto value : unpackOp.getVals()) {
                  addUseAfter(value, op, notCounted);
               }
            })
            .Case<util::GetTupleOp>([&](util::GetTupleOp getTupleOp) {
               addUseAfter(getTupleOp.getVal(), op, notCounted);
            })
            .Case<db::ListSetOp>([&](db::ListSetOp setOp) {
               addUse(setOp.getElement(), op, notCounted);
            })
            .Case<db::AsNullableOp>([&](db::AsNullableOp asNullableOp) {
               addUse(asNullableOp.getVal(), op, notCounted);
            })
            .Case<util::PackOp>([&](util::PackOp packOp) {
               for (auto value : packOp.getVals()) {
                  addUse(value, op, notCounted);
               }
            })
            .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp selectOp) {
               // arith.select can't grow ref counts; rewrite as scf.if and bump in each arm.
               if (!typeNeedsManagement(selectOp.getType())) return;
               mlir::OpBuilder builder(selectOp);
               auto ifOp = builder.create<mlir::scf::IfOp>(selectOp->getLoc(), selectOp.getType(), selectOp.getCondition(), true);
               {
                  mlir::OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
                  thenBuilder.create<db::MemoryAddUse>(selectOp->getLoc(), selectOp.getTrueValue());
                  thenBuilder.create<mlir::scf::YieldOp>(selectOp->getLoc(), selectOp.getTrueValue());
               }
               {
                  mlir::OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
                  elseBuilder.create<db::MemoryAddUse>(selectOp->getLoc(), selectOp.getFalseValue());
                  elseBuilder.create<mlir::scf::YieldOp>(selectOp->getLoc(), selectOp.getFalseValue());
               }
               selectOp.replaceAllUsesWith(ifOp.getResult(0));
               if (returnedValues.contains(selectOp.getResult())) {
                  returnedValues.erase(selectOp.getResult());
                  returnedValues.insert(ifOp.getResult(0));
               }
               selectOp.erase();
               op = ifOp;
            })
            .Default([&](mlir::Operation*) {});
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
            if (!typeNeedsManagement(operand.get().getType())) continue;
            if (notCounted.contains(operand.get())) continue;
            mlir::OpBuilder builder(block->getTerminator());
            mlir::Value newVal = builder.create<db::MemoryPromoteToGlobal>(
               block->getTerminator()->getLoc(), operand.get().getType(), operand.get());
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
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::subop::createMemoryMgmtPass() {
   return std::make_unique<MemoryMgmtPass>();
}
