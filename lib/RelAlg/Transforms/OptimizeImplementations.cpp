#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class OptimizeImplementations : public mlir::PassWrapper<OptimizeImplementations, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-optimize-implementations"; }

   public:
   bool hashImplPossible(mlir::Block* block, mlir::relalg::ColumnSet availableLeft, mlir::relalg::ColumnSet availableRight) { //todo: does not work always
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      mlir::relalg::ColumnSet leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      bool res = false;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.attr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred() && mlir::relalg::HashJoinUtils::isAndedResult(op)) {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  res = true;
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  res = true;
               }
            }
         } else {
            mlir::relalg::ColumnSet attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return res;
   }
   void runOnOperation() override {
      getOperation().walk([&](Operator op) {
         ::llvm::TypeSwitch<mlir::Operation*, void>(op.getOperation())
            .Case<mlir::relalg::InnerJoinOp, mlir::relalg::MarkJoinOp,mlir::relalg::CollectionJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
               }
            })
            .Case<mlir::relalg::SemiJoinOp, mlir::relalg::AntiSemiJoinOp, mlir::relalg::OuterJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  if (left->hasAttr("rows") && right->hasAttr("rows")) {
                     double rowsLeft = 0;
                     double rowsRight = 0;
                     if (auto lDAttr = left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                        rowsLeft = lDAttr.getValueAsDouble();
                     } else if (auto lIAttr = left->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                        rowsLeft = lIAttr.getInt();
                     }
                     if (auto rDAttr = right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                        rowsRight = rDAttr.getValueAsDouble();
                     } else if (auto rIAttr = right->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                        rowsRight = rIAttr.getInt();
                     }
                     if (rowsLeft < rowsRight) {
                        op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "markhash"));
                     } else {
                        op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                     }
                  } else {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                  }
               }
            })
            .Case<mlir::relalg::SingleJoinOp>([&](mlir::relalg::SingleJoinOp op) {
               if (auto returnOp = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(op.getPredicateBlock().getTerminator())) {
                  if (returnOp.results().empty()) {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "constant"));
                  }
               }
               auto left = mlir::cast<Operator>(op.leftChild());
               auto right = mlir::cast<Operator>(op.rightChild());
               if (hashImplPossible(&op.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
               }
            })

            .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) {})
            .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) {})

            .Default([&](auto x) {
            });
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createOptimizeImplementationsPass() { return std::make_unique<OptimizeImplementations>(); }
} // end namespace relalg
} // end namespace mlir