
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>

namespace {
class ExtractNestedOperators : public mlir::PassWrapper<ExtractNestedOperators, mlir::FunctionPass> {
   public:
   void sanitizeOp(Operator& innerOperator, TupleLamdaOperator moveBefore, mlir::BlockAndValueMapping& mapping, mlir::Operation* op) const {
      for (size_t i = 0; i < op->getNumOperands(); i++) {
         mlir::Value v = op->getOperand(i);
         if (mapping.contains(v)) {
            op->setOperand(i, mapping.lookup(v));
            continue;
         }
         if (innerOperator->getRegion(0).isAncestor(v.getParentRegion())) {
            continue;
         }
         mlir::Operation* definingOp = v.getDefiningOp();
         if (definingOp && moveBefore->isAncestor(definingOp)) {
            mlir::OpBuilder builder(op);
            auto* clonedOp = builder.clone(*definingOp, mapping);
            clonedOp->moveBefore(op);
            for (size_t i = 0; i < op->getNumResults(); i++) {
               definingOp->getResult(i).replaceUsesWithIf(clonedOp->getResult(i), [innerOperator](mlir::OpOperand& operand) {
                  return operand.getOwner()->getParentOfType<Operator>() == innerOperator;
               });
            }
            for (size_t i = 0; i < definingOp->getNumResults(); i++) {
               mapping.map(definingOp->getResult(i), clonedOp->getResult(i));
            }
            sanitizeOp(innerOperator, moveBefore, mapping, clonedOp);
         }
      }
   }
   void runOnFunction() override {
      getFunction().walk([&](Operator innerOperator) {
         if (innerOperator->getParentOfType<TupleLamdaOperator>()) {
            TupleLamdaOperator o = innerOperator->getParentOfType<TupleLamdaOperator>();
            mlir::BlockAndValueMapping mapping;
            TupleLamdaOperator toMoveBefore;

            while (o) {
               if (auto innerLambda = mlir::dyn_cast_or_null<TupleLamdaOperator>(innerOperator.getOperation())) {
                  mapping.map(o.getLambdaArgument(), innerLambda.getLambdaArgument());
               }
               toMoveBefore = o;
               o = o->getParentOfType<TupleLamdaOperator>();
            }
            innerOperator->walk([&](mlir::Operation* op) {
               if (!mlir::isa<Operator>(op)) {
                  sanitizeOp(innerOperator, toMoveBefore, mapping, op);
               }
            });
            innerOperator->moveBefore(toMoveBefore);
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createExtractNestedOperatorsPass() { return std::make_unique<ExtractNestedOperators>(); }
} // end namespace relalg
} // end namespace mlir