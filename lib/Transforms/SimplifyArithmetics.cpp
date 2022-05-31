#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <iostream>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
#include "CustomCanonicalization.inc"

//Pattern that optimizes the join order
class SimplifyArithmetics : public mlir::PassWrapper<SimplifyArithmetics, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "simplify-arithmetics"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<ConstAddCmpFold>(&getContext());
         patterns.insert<ConstMulCmpFold>(&getContext());

         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir {

std::unique_ptr<Pass> createSimplifyArithmeticsPass() { return std::make_unique<SimplifyArithmetics>(); }

} // end namespace mlir