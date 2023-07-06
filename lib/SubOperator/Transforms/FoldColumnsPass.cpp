#include "llvm/Support/Debug.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>
namespace {

class PushRenamingUp : public mlir::RewritePattern {
   public:
   PushRenamingUp(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::RenamingOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto renamingOp = mlir::cast<mlir::subop::RenamingOp>(op);
      auto loc=op->getLoc();
      if (!renamingOp->hasOneUse()) return mlir::failure();
      auto columns = renamingOp.getColumns();

      auto* user = *renamingOp->getUsers().begin();
      if (auto columnFoldable = mlir::dyn_cast_or_null<mlir::subop::ColumnFoldable>(user)) {
         mlir::subop::ColumnFoldInfo columnFoldInfo;
         for (auto c : renamingOp.getColumns()) {
            auto* newColumn = &c.cast<mlir::tuples::ColumnDefAttr>().getColumn();
            auto* prevColumn = &c.cast<mlir::tuples::ColumnDefAttr>().getFromExisting().cast<mlir::ArrayAttr>()[0].cast<mlir::tuples::ColumnRefAttr>().getColumn();
            columnFoldInfo.directMappings[newColumn] = prevColumn;
         }
         if (columnFoldable.foldColumns(columnFoldInfo).succeeded()) {
            rewriter.replaceOp(op, renamingOp.getStream());
            if (user->getNumResults() == 1) {
               rewriter.setInsertionPointAfter(columnFoldable);
               auto renamed = rewriter.create<mlir::subop::RenamingOp>(loc, user->getResult(0), columns);
               user->getResult(0).replaceAllUsesExcept(renamed, renamed.getOperation());
            }
            return mlir::success();
         }
      }
      return mlir::failure();
   }
};
class FoldColumnsPass : public mlir::PassWrapper<FoldColumnsPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldColumnsPass)
   virtual llvm::StringRef getArgument() const override { return "subop-fold-columns"; }

   void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<PushRenamingUp>(&getContext());
      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::subop::createFoldColumnsPass() { return std::make_unique<FoldColumnsPass>(); }