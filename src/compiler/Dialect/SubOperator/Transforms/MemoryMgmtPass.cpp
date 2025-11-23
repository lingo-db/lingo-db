#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/helper.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/ADT/TypeSwitch.h>

#include <unordered_set>
#include <mlir/Dialect/SCF/IR/SCF.h>
namespace lingodb::compiler::dialect::util {
class UnPackOp;
}
namespace {
using namespace lingodb::compiler::dialect;

class MemoryMgmtPass : public mlir::PassWrapper<MemoryMgmtPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryMgmtPass)
   virtual llvm::StringRef getArgument() const override { return "subop-memory-mgmt"; }
   bool typeNeedsManagement(mlir::Type t) {
      if (auto nullableType = mlir::dyn_cast<db::NullableType>(t)) {
         t = nullableType.getType();
      }
      if (mlir::isa<db::StringType, db::ListType>(t)) {
         return true;
      }
      if (mlir::isa<mlir::TupleType>(t)) {
         auto tupleType = mlir::cast<mlir::TupleType>(t);
         for (auto elementType : tupleType.getTypes()) {
            if (typeNeedsManagement(elementType)) {
               return true;
            }
         }
      }
      return false;
   }
   void addUse(mlir::Value val, mlir::Operation* insertBeforeOp) {
      if (typeNeedsManagement(val.getType())) {
         if (mlir::isa<mlir::TupleType>(val.getType())) {
            mlir::OpBuilder builder(insertBeforeOp);
            llvm::SmallVector<mlir::Value> unpacked;
            builder.createOrFold<util::UnPackOp>(unpacked, insertBeforeOp->getLoc(), val);
            for (auto element : unpacked) {
               addUse(element, insertBeforeOp);
            }

         } else {
            mlir::OpBuilder builder(insertBeforeOp);
            builder.create<db::MemoryAddUse>(insertBeforeOp->getLoc(), val);
         }
      }
   }
   void addUseAfter(mlir::Value val, mlir::Operation* insertAfterOp) {
      if (typeNeedsManagement(val.getType())) {
         if (mlir::isa<mlir::TupleType>(val.getType())) {
            mlir::OpBuilder builder(insertAfterOp->getContext());
            builder.setInsertionPointAfter(insertAfterOp);
            llvm::SmallVector<mlir::Value> unpacked;
            builder.createOrFold<util::UnPackOp>(unpacked, insertAfterOp->getLoc(), val);
            for (auto element : unpacked) {
               addUseAfter(element, insertAfterOp);
            }

         } else {
            mlir::OpBuilder builder(insertAfterOp->getContext());
            builder.setInsertionPointAfter(insertAfterOp);
            builder.create<db::MemoryAddUse>(insertAfterOp->getLoc(), val);
         }
      }
   }
   void cleanupUse(mlir::Operation* insertBeforeOp, mlir::Value val) {
      if (auto tupleType = mlir::dyn_cast<mlir::TupleType>(val.getType())) {
         // for tuples, we need to cleanup each element
         mlir::OpBuilder builder(insertBeforeOp);
         llvm::SmallVector<mlir::Value> unpacked;
         builder.createOrFold<util::UnPackOp>(unpacked, insertBeforeOp->getLoc(), val);
         for (auto element : unpacked) {
            if (typeNeedsManagement(element.getType())) {
               cleanupUse(insertBeforeOp, element);
            }
         }
         return;
      }
      if (typeNeedsManagement(val.getType())) {
         mlir::OpBuilder builder(insertBeforeOp);
         if (auto listType = mlir::dyn_cast<db::ListType>(val.getType())) {
            if (mlir::isa<db::StringType>(listType.getElementType())) {
               auto loc = insertBeforeOp->getLoc();
               std::string name = "_cleanup_list_str";
               auto moduleOp = insertBeforeOp->getParentOfType<mlir::ModuleOp>();
               mlir::func::FuncOp cleanupFn = moduleOp.lookupSymbol<mlir::func::FuncOp>(name);
               auto fnType = builder.getFunctionType({listType}, {});

               if (!cleanupFn) {
                  mlir::OpBuilder::InsertionGuard guard(builder);
                  builder.setInsertionPointToStart(insertBeforeOp->getParentOfType<mlir::ModuleOp>().getBody());
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
   }
   void runOnOperation() override {
      // Walk the module and process every subop::MapOp.
      // Currently we iterate the MapOp's regions and blocks so we can
      // inspect and later transform per-block nested code (placeholder).
      auto module = getOperation();
      module.walk([&](subop::MapOp mapOp) {
         // Iterate all regions and blocks inside the MapOp.
         mapOp->walk([&](mlir::Block* block) {
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
                        addUse(arg, op);
                     }
                  })
                  .Case<mlir::scf::WhileOp>([&](mlir::scf::WhileOp whileOp) {
                     for (auto arg : whileOp.getInits()) {
                        addUse(arg, op);
                     }
                  })
                  .Case<db::ListAppendOp>([&](db::ListAppendOp appendOp) {
                     addUse(appendOp.getElement(), op);
                  })
                  .Case<db::ListGetOp>([&](db::ListGetOp listGetOp) {
                     addUseAfter(listGetOp.getElement(), op);
                  })
                  .Case<util::UnPackOp>([&](util::UnPackOp unpackOp) {
                     for (auto value : unpackOp.getVals()) {
                        addUseAfter(value, op);
                     }
                  })
                  .Case<util::GetTupleOp>([&](util::GetTupleOp getTupleOp) {
                     addUseAfter(getTupleOp.getVal(), op);
                  })
                  .Case<db::ListSetOp>([&](db::ListSetOp setOp) {
                     addUse(setOp.getElement(), op);
                  })
                  .Case<util::PackOp>([&](util::PackOp packOp) {
                     for (auto value : packOp.getVals()) {
                        addUse(value, op);
                     }
                  })
                  .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp selectOp) {
                     // if result is managed: replace with IfOp and add use in each branch
                     mlir::OpBuilder builder(selectOp);
                     if (typeNeedsManagement(selectOp.getType())) {
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
                        //also replace in returnedValues
                        if (returnedValues.contains(selectOp.getResult())) {
                           returnedValues.erase(selectOp.getResult());
                           returnedValues.insert(ifOp.getResult(0));
                        }
                        selectOp.erase();
                        op = ifOp; // continue processing the IfOp
                     }
                  })
                  .Default([&](mlir::Operation*) {
                     // Do nothing
                  });
               for (auto result : op->getResults()) {
                  if (typeNeedsManagement(result.getType())) {
                     if (returnedValues.contains(result)) {
                        returnedValues.erase(result); // value is from this block -> no need for extra treatment
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
               cleanupUse(block->getTerminator(), value);
            }
            if (block == &mapOp.getFn().front()) {
               for (auto& operand : terminator->getOpOperands()) {
                  if (typeNeedsManagement(operand.get().getType())) {
                     mlir::OpBuilder builder(block->getTerminator());
                     mlir::Value newVal = builder.create<db::MemoryPromoteToGlobal>(block->getTerminator()->getLoc(), operand.get().getType(),operand.get());
                     operand.set(newVal);
                  }
               }
            } else {
               for (auto value : returnedValues) {
                  if (typeNeedsManagement(value.getType())) {
                     mlir::OpBuilder builder(block, block->getTerminator()->getIterator());
                     addUse(value, block->getTerminator());
                  }
               }
            }
         });
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> subop::createMemoryMgmtPass() { return std::make_unique<MemoryMgmtPass>(); }