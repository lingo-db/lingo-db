#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGFUSEMATMUL
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

class GraphAlgFuseMatMul
   : public impl::GraphAlgFuseMatMulBase<GraphAlgFuseMatMul> {
   using impl::GraphAlgFuseMatMulBase<GraphAlgFuseMatMul>::GraphAlgFuseMatMulBase;

   void runOnOperation() final;
};

} // namespace

// Flatten an `ewise ADD` tree rooted at `v` into its addend leaves: an operand
// whose defining op is `ewise` with op == ADD is recursed into, everything else
// is a leaf. ADD is commutative and associative in every semiring used here, so
// the leaf order carries no meaning.
static void flattenAddTree(mlir::Value v,
                           llvm::SmallVectorImpl<mlir::Value>& addends) {
   if (auto ew = v.getDefiningOp<ElementWiseOp>()) {
      if (ew.getOp() == BinaryOp::ADD) {
         flattenAddTree(ew.getLhs(), addends);
         flattenAddTree(ew.getRhs(), addends);
         return;
      }
   }
   addends.push_back(v);
}

// Semiring distributivity: mxm(A, X) (+) mxm(B, X) == mxm(A (+) B, X). Within an
// `ewise ADD` tree, fuse all MatMul addends sharing the same rhs into a single
// MatMul whose lhs is the (+)-combination of the individual lhs matrices. For
// undirected propagation (e.g. WCC computes A·X + Aᵀ·X each iteration) this
// collapses the two matmuls — and their downstream reduce/pick_any — into one,
// and the fused A (+) Aᵀ is loop-invariant so HoistInvariantStatePass builds its
// edge index once instead of twice.
static mlir::LogicalResult fuseMatMulInAddTree(ElementWiseOp op,
                                               mlir::PatternRewriter& rewriter) {
   if (op.getOp() != BinaryOp::ADD) return mlir::failure();

   // Only rewrite at the root of an ADD tree, so each tree is rebuilt once.
   for (auto* user : op->getUsers())
      if (auto ew = llvm::dyn_cast<ElementWiseOp>(user))
         if (ew.getOp() == BinaryOp::ADD)
            return mlir::failure();

   llvm::SmallVector<mlir::Value> addends;
   flattenAddTree(op.getResult(), addends);

   // Group MatMul addends by their (shared) rhs operand, preserving first-seen
   // order; non-MatMul addends pass through unchanged.
   llvm::MapVector<mlir::Value, llvm::SmallVector<MatMulOp>> mxmByRhs;
   llvm::SmallVector<mlir::Value> others;
   for (mlir::Value v : addends) {
      if (auto mm = v.getDefiningOp<MatMulOp>()) {
         mxmByRhs[mm.getRhs()].push_back(mm);
      } else {
         others.push_back(v);
      }
   }

   bool fuseable = false;
   for (auto& group : mxmByRhs)
      if (group.second.size() >= 2) { fuseable = true; break; }
   if (!fuseable) return mlir::failure();

   auto loc = op.getLoc();
   rewriter.setInsertionPoint(op);

   llvm::SmallVector<mlir::Value> newAddends(others.begin(), others.end());
   for (auto& group : mxmByRhs) {
      auto& mxms = group.second;
      mlir::Value rhs = group.first;
      // A fused ewise needs matching lhs types (AllTypesMatch). Equal result and
      // rhs types already force this for two matmuls, but guard defensively:
      // any lhs whose type differs from the first is left as its own addend.
      mlir::Value fusedLhs = mxms[0].getLhs();
      llvm::SmallVector<MatMulOp> unfused;
      for (size_t i = 1; i < mxms.size(); ++i) {
         if (mxms[i].getLhs().getType() != fusedLhs.getType()) {
            unfused.push_back(mxms[i]);
            continue;
         }
         fusedLhs = rewriter.create<ElementWiseOp>(loc, fusedLhs, BinaryOp::ADD,
                                                   mxms[i].getLhs());
      }
      if (fusedLhs == mxms[0].getLhs() && unfused.size() == mxms.size() - 1) {
         // Nothing actually fused for this group (all type-mismatched) — keep
         // every matmul as-is.
         for (MatMulOp mm : mxms) newAddends.push_back(mm.getResult());
      } else {
         newAddends.push_back(
            rewriter.create<MatMulOp>(loc, fusedLhs, rhs).getResult());
         for (MatMulOp mm : unfused) newAddends.push_back(mm.getResult());
      }
   }

   mlir::Value result = newAddends[0];
   for (size_t i = 1; i < newAddends.size(); ++i)
      result = rewriter.create<ElementWiseOp>(loc, result, BinaryOp::ADD,
                                              newAddends[i]);

   rewriter.replaceOp(op, result);
   return mlir::success();
}

void GraphAlgFuseMatMul::runOnOperation() {
   mlir::RewritePatternSet patterns(&getContext());
   patterns.add(fuseMatMulInAddTree);

   if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
   }
}

} // namespace graphalg
