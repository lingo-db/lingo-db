#include "llvm/Support/Debug.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>
namespace {
using namespace lingodb::compiler::dialect;

class PushRenamingUp : public mlir::RewritePattern {
   public:
   PushRenamingUp(mlir::MLIRContext* context)
      : RewritePattern(subop::RenamingOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto renamingOp = mlir::cast<subop::RenamingOp>(op);
      auto loc = op->getLoc();
      if (!renamingOp->hasOneUse()) return mlir::failure();
      auto columns = renamingOp.getColumns();

      auto* user = *renamingOp->getUsers().begin();
      if (auto columnFoldable = mlir::dyn_cast_or_null<subop::ColumnFoldable>(user)) {
         subop::ColumnMapping columnFoldInfo;
         for (auto c : renamingOp.getColumns()) {
            auto* newColumn = &mlir::cast<tuples::ColumnDefAttr>(c).getColumn();
            auto* prevColumn = &mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(mlir::cast<tuples::ColumnDefAttr>(c).getFromExisting())[0]).getColumn();
            columnFoldInfo.mapRaw(newColumn, prevColumn);
         }
         if (columnFoldable.foldColumns(columnFoldInfo).succeeded()) {
            rewriter.replaceOp(op, renamingOp.getStream());
            if (user->getNumResults() == 1) {
               rewriter.setInsertionPointAfter(columnFoldable);
               auto renamed = rewriter.create<subop::RenamingOp>(loc, user->getResult(0), columns);
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
      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> subop::createFoldColumnsPass() { return std::make_unique<FoldColumnsPass>(); }