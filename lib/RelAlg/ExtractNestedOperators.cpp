
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
   void sanitizeOp(Operator& inner_operator, mlir::BlockAndValueMapping& mapping, mlir::Operation* op) const {
      for (size_t i = 0; i < op->getNumOperands(); i++) {
         mlir::Value v = op->getOperand(i);
         if (mapping.contains(v)) {
            op->setOperand(i, mapping.lookup(v));
            continue;
         }
         if (inner_operator->getRegion(0).isAncestor(v.getParentRegion())) {
            continue;
         }
         mlir::Operation* definingOp = v.getDefiningOp();
         if (definingOp && definingOp->getParentOfType<Operator>()) {
            mlir::OpBuilder builder(op);
            auto cloned_op = builder.clone(*definingOp, mapping);
            cloned_op->moveBefore(op);
            for (size_t i = 0; i < op->getNumResults(); i++) {
               definingOp->getResult(i).replaceUsesWithIf(cloned_op->getResult(i), [&](mlir::OpOperand& operand) {
                  return operand.getOwner()->getParentOfType<Operator>() == inner_operator;
               });
            }
            for (size_t i = 0; i < definingOp->getNumResults(); i++) {
               mapping.map(definingOp->getResult(i), cloned_op->getResult(i));
            }
            sanitizeOp(inner_operator, mapping, cloned_op);
         }
      }
   }
   void runOnFunction() override {
      getFunction().walk([&](Operator inner_operator) {
         if (inner_operator->getParentOfType<TupleLamdaOperator>()) {
            TupleLamdaOperator o = inner_operator->getParentOfType<TupleLamdaOperator>();
            mlir::BlockAndValueMapping mapping;

            while (o) {
               if (inner_operator->getNumRegions() == 1 && !inner_operator->getRegion(0).empty() && !(inner_operator->getRegion(0).front().getNumArguments() > 1)) {
                  mapping.map(o.getLambdaBlock().getArgument(0), inner_operator->getRegion(0).front().getArgument(0));
               }
               o = o->getParentOfType<TupleLamdaOperator>();
            }
            inner_operator->walk([&](mlir::Operation* op) {
               if (!mlir::isa<Operator>(op)) {
                  sanitizeOp(inner_operator, mapping, op);
               }
            });
            inner_operator->moveBefore(inner_operator->getParentOfType<Operator>());
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