#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
using namespace lingodb::compiler::dialect;

/// Flatten the value expression under the join predicate's return into AND leaves
static void collectAndLeaves(mlir::Value v, llvm::SmallVectorImpl<mlir::Value>& leaves) {
   if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(v.getDefiningOp())) {
      for (mlir::Value operand : andOp->getOperands()) {
         collectAndLeaves(operand, leaves);
      }
      return;
   }
   leaves.push_back(v);
}

/// From the join predicate region, collect columns that appear on the right input for each left–right equality (hash join key on the right).
static relalg::ColumnSet extractRightJoinKeyColumns(relalg::OuterJoinOp outerJoin, relalg::AvailabilityCache& cache) {
   relalg::ColumnSet outRightKeys;
   auto* term = outerJoin.getPredicateBlock().getTerminator();
   auto returnOp = mlir::dyn_cast<tuples::ReturnOp>(term);
   if (!returnOp || returnOp.getNumOperands() == 0) {
      return outRightKeys;
   }

   llvm::SmallVector<mlir::Value, 8> leaves;
   collectAndLeaves(returnOp.getOperand(0), leaves);
   if (leaves.empty()) {
      return outRightKeys;
   }

   auto left = mlir::cast<Operator>(outerJoin.getLeft().getDefiningOp());
   auto right = mlir::cast<Operator>(outerJoin.getRight().getDefiningOp());
   auto availableLeft = left.getAvailableColumns(cache);
   auto availableRight = right.getAvailableColumns(cache);

   outRightKeys = relalg::ColumnSet();
   for (mlir::Value leaf : leaves) {
      auto* defOp = leaf.getDefiningOp();
      auto cmpOp = mlir::dyn_cast_or_null<relalg::CmpOpInterface>(defOp);
      if (!cmpOp || !cmpOp.isEqualityPred(false)) {
         continue;
      }
      auto getLeftCol = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getLeft().getDefiningOp());
      auto getRightCol = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getRight().getDefiningOp());
      if (!getLeftCol || !getRightCol) {
         continue;
      }
      relalg::ColumnSet leftCols = relalg::ColumnSet::from(getLeftCol.getAttr());
      relalg::ColumnSet rightCols = relalg::ColumnSet::from(getRightCol.getAttr());

      if (leftCols.isSubsetOf(availableLeft) && rightCols.isSubsetOf(availableRight)) {
         outRightKeys.insert(rightCols);
      } else if (leftCols.isSubsetOf(availableRight) && rightCols.isSubsetOf(availableLeft)) {
         outRightKeys.insert(leftCols);
      }
   }
   return outRightKeys;
}

class EliminateTrivialJoin final : public mlir::OpRewritePattern<relalg::OuterJoinOp> {
   public:
   using mlir::OpRewritePattern<relalg::OuterJoinOp>::OpRewritePattern;

   mlir::LogicalResult matchAndRewrite(relalg::OuterJoinOp outerJoinOp, mlir::PatternRewriter& rewriter) const override {
      if (!outerJoinOp.getMapping().empty()) {
         return mlir::failure();
      }

      auto right = mlir::dyn_cast<Operator>(outerJoinOp.getRight().getDefiningOp());

      relalg::AvailabilityCache cache;
      auto rightKeyCols = extractRightJoinKeyColumns(outerJoinOp, cache);

      auto fds = right.getFDs();
      if (rightKeyCols.empty() || !fds.isDuplicateFreeKey(rightKeyCols)) {
         return mlir::failure();
      }

      rewriter.replaceOp(outerJoinOp, outerJoinOp.getLeft());
      return mlir::success();
   }
};

class EliminateTrivialJoinPass
   : public mlir::PassWrapper<EliminateTrivialJoinPass, mlir::OperationPass<mlir::func::FuncOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateTrivialJoinPass)
   llvm::StringRef getArgument() const override { return "relalg-eliminate-outerjoin-empty-right"; }

   void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<EliminateTrivialJoin>(&getContext());
      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};

} // namespace

std::unique_ptr<mlir::Pass> relalg::createEliminateTrivialJoinPass() {
   return std::make_unique<EliminateTrivialJoinPass>();
}
