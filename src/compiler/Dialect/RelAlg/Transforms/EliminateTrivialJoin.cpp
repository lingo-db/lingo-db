#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
using namespace lingodb::compiler::dialect;

class EliminateTrivialJoin final : public mlir::OpRewritePattern<relalg::OuterJoinOp> {
   public:
   using mlir::OpRewritePattern<relalg::OuterJoinOp>::OpRewritePattern;

   mlir::LogicalResult matchAndRewrite(relalg::OuterJoinOp outerJoinOp, mlir::PatternRewriter& rewriter) const override {
      bool useHash = outerJoinOp->hasAttr("useHashJoin");
      if (!useHash) {
         return mlir::failure();
      }
      auto nullsEqual = outerJoinOp->getAttrOfType<mlir::ArrayAttr>("nullsEqual");
      for (auto nE : nullsEqual) {
         bool nullequal = mlir::cast<mlir::IntegerAttr>(nE).getInt();
         if (nullequal) {
            return mlir::failure();
         }
      }

      auto right = mlir::dyn_cast<Operator>(outerJoinOp.getRight().getDefiningOp());
      if (!outerJoinOp->hasAttr("rightHash")) {
         return mlir::failure();
      }

      auto rightKeysAttr = mlir::cast<mlir::ArrayAttr>(outerJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash"));
      auto rightKeyCols = relalg::ColumnSet::fromArrayAttr(rightKeysAttr);

      auto fds = right.getFDs();
      if (!fds.isDuplicateFreeKey(rightKeyCols)) {
         return mlir::failure();
      }
      if (!outerJoinOp.getMapping().empty()) {
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
