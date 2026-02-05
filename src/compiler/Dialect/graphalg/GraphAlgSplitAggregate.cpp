#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGSPLITAGGREGATE
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

class GraphAlgSplitAggregate
   : public impl::GraphAlgSplitAggregateBase<GraphAlgSplitAggregate> {
   using impl::GraphAlgSplitAggregateBase<
      GraphAlgSplitAggregate>::GraphAlgSplitAggregateBase;

   void runOnOperation() final;
};

} // namespace

// Split MatMulOp into MatMulJoinOp + DeferredReduceOp
static mlir::LogicalResult decomposeMatMul(MatMulOp op,
                                           mlir::PatternRewriter& rewriter) {
   auto joinOp = rewriter.create<MatMulJoinOp>(op.getLoc(), op.getType(),
                                               op.getLhs(), op.getRhs());

   auto innerDim = op.getLhs().getType().getCols();
   assert(innerDim == op.getRhs().getType().getRows());
   if (innerDim.isOne()) {
      // Never produces more than one value per cell, so we can omit the
      // DeferredReduceOp.
      rewriter.replaceOp(op, joinOp);
      return mlir::success();
   }

   rewriter.replaceOpWithNewOp<DeferredReduceOp>(op, op.getType(),
                                                 mlir::ValueRange{joinOp});
   return mlir::success();
}

// Lower ElementWiseAdd into DeferredReduceOp
static mlir::LogicalResult
decomposeElementWiseAdd(ElementWiseAddOp op, mlir::PatternRewriter& rewriter) {
   rewriter.replaceOpWithNewOp<DeferredReduceOp>(
      op, op.getType(), mlir::ValueRange{op.getLhs(), op.getRhs()});
   return mlir::success();
}

// Rewrite ReduceOp into DeferredReduceOp so we only need to write lowering for
// the latter.
static mlir::LogicalResult deferReduce(ReduceOp op,
                                       mlir::PatternRewriter& rewriter) {
   rewriter.replaceOpWithNewOp<DeferredReduceOp>(op, op.getType(),
                                                 op.getInput());
   return mlir::success();
}

static mlir::LogicalResult convertMakeDense(MakeDenseOp op,
                                            mlir::PatternRewriter& rewriter) {
   auto resultType = op.getType();
   auto sring = llvm::cast<SemiringTypeInterface>(resultType.getSemiring());
   auto constantOp = rewriter.create<ConstantMatrixOp>(op.getLoc(), resultType,
                                                       sring.addIdentity());
   rewriter.replaceOpWithNewOp<ElementWiseAddOp>(op, op.getInput(), constantOp);
   return mlir::success();
}

void GraphAlgSplitAggregate::runOnOperation() {
   mlir::RewritePatternSet patterns(&getContext());
   patterns.add(decomposeMatMul);
   patterns.add(decomposeElementWiseAdd);
   patterns.add(deferReduce);
   patterns.add(convertMakeDense);

   if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
   }
}

} // namespace graphalg
