#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class ExtractNestedOperators : public mlir::PassWrapper<ExtractNestedOperators, mlir::OperationPass<mlir::func::FuncOp>> {
   public:
   virtual llvm::StringRef getArgument() const override { return "relalg-extract-nested-operators"; }

   void sanitizeOp(mlir::BlockAndValueMapping& mapping, mlir::Operation* op) const {
      for (size_t i = 0; i < op->getNumOperands(); i++) {
         mlir::Value v = op->getOperand(i);
         if (mapping.contains(v)) {
            op->setOperand(i, mapping.lookup(v));
            continue;
         }
      }
   }
   void runOnOperation() override {
      getOperation().walk([&](Operator innerOperator) {
         if (auto o = mlir::dyn_cast_or_null<TupleLamdaOperator>(innerOperator->getParentOp())) {
            mlir::BlockAndValueMapping mapping;
            TupleLamdaOperator toMoveBefore;
            while (o) {
               if (auto innerLambda = mlir::dyn_cast_or_null<TupleLamdaOperator>(innerOperator.getOperation())) {
                  mapping.map(o.getLambdaArgument(), innerLambda.getLambdaArgument());
               }
               toMoveBefore = o;
               o = mlir::dyn_cast_or_null<TupleLamdaOperator>(o->getParentOp());
            }
            innerOperator->walk([&](mlir::Operation* op) {
               if (!mlir::isa<Operator>(op)) {
                  mlir::relalg::detail::inlineOpIntoBlock(op, toMoveBefore, innerOperator, op->getBlock(), mapping);
                  sanitizeOp(mapping, op);
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