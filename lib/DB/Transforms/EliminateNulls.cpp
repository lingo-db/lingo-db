#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/DB/IR/DBOps.h"

#include <iostream>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
#include "EliminateNulls.inc"

//Pattern that optimizes the join order
class EliminateNulls : public mlir::PassWrapper<EliminateNulls, mlir::OperationPass<mlir::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "eliminate-nulls"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<EliminateNullCmp>(&getContext());
         patterns.insert<EliminateNullAdd>(&getContext());
         patterns.insert<EliminateIsNull>(&getContext());
         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir::db{

std::unique_ptr<Pass> createEliminateNullsPass() { return std::make_unique<EliminateNulls>(); }

} // end namespace mlir::db