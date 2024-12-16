#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
using namespace lingodb::compiler::dialect;

class ExtractNestedOperators : public mlir::PassWrapper<ExtractNestedOperators, mlir::OperationPass<mlir::func::FuncOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractNestedOperators)
   virtual llvm::StringRef getArgument() const override { return "relalg-extract-nested-operators"; }

   void sanitizeOp(mlir::IRMapping& mapping, mlir::Operation* op) const {
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
         if (auto o = mlir::dyn_cast_or_null<TupleLamdaOperator>(innerOperator->getParentOfType<Operator>().getOperation())) {
            mlir::IRMapping mapping;
            TupleLamdaOperator toMoveBefore;
            while (o) {
               if (auto innerLambda = mlir::dyn_cast_or_null<TupleLamdaOperator>(innerOperator.getOperation())) {
                  mapping.map(o.getLambdaArgument(), innerLambda.getLambdaArgument());
               }
               toMoveBefore = o;
               o = mlir::dyn_cast_or_null<TupleLamdaOperator>(o->getParentOfType<Operator>().getOperation());
            }
            innerOperator->walk([&](mlir::Operation* op) {
               if (!mlir::isa<Operator>(op)&&op->getParentOp()==innerOperator.getOperation()) {
                  relalg::detail::inlineOpIntoBlock(op, toMoveBefore, op->getBlock(), mapping);
                  sanitizeOp(mapping, op);
               }
            });
            innerOperator->moveBefore(toMoveBefore);
         }
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createExtractNestedOperatorsPass() { return std::make_unique<ExtractNestedOperators>(); }
