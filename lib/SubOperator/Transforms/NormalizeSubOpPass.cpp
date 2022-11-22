#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {
static std::pair<mlir::tuples::ColumnDefAttr, mlir::tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   mlir::tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}
class SplitTableScan : public mlir::RewritePattern {
   public:
   SplitTableScan(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::ScanOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto scanOp = mlir::cast<mlir::subop::ScanOp>(op);
      if (!scanOp.getState().getType().isa<mlir::subop::TableType>()) return mlir::failure();
      auto tableType = scanOp.getState().getType().cast<mlir::subop::TableType>();
      std::vector<mlir::Attribute> memberNames;
      std::vector<mlir::Attribute> memberTypes;
      auto tableMembers = tableType.getMembers();
      for (auto i = 0ul; i < tableMembers.getTypes().size(); i++) {
         auto type = tableMembers.getTypes()[i].cast<mlir::TypeAttr>();
         auto name = tableMembers.getNames()[i].cast<mlir::StringAttr>();
         if (scanOp.getMapping().contains(name)) {
            memberNames.push_back(name);
            memberTypes.push_back(type);
         }
      }
      auto members = mlir::subop::StateMembersAttr::get(getContext(), mlir::ArrayAttr::get(getContext(), memberNames), mlir::ArrayAttr::get(getContext(), memberTypes));

      auto [refDef, refRef] = createColumn(mlir::subop::TableEntryRefType::get(getContext(), members), "scan", "ref");
      mlir::Value scanRefsOp = rewriter.create<mlir::subop::ScanRefsOp>(op->getLoc(), scanOp.getState(), refDef);
      rewriter.replaceOpWithNewOp<mlir::subop::GatherOp>(op, scanRefsOp, refRef, scanOp.getMapping());
      return mlir::success();
   }
};
class SplitGenericScan : public mlir::RewritePattern {
   public:
   SplitGenericScan(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::ScanOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto scanOp = mlir::cast<mlir::subop::ScanOp>(op);
      if (scanOp.getState().getType().isa<mlir::subop::TableType>()) return mlir::failure();
      if (scanOp.getState().getType().isa<mlir::subop::HashMapType>()) return mlir::failure();
      //if (scanOp.getState().getType().isa<mlir::subop::LazyMultiMapType>()) return mlir::failure();
      //todo: check that one can obtain references
      auto [refDef, refRef] = createColumn(mlir::subop::EntryRefType::get(getContext(), scanOp.getState().getType()), "scan", "ref");
      mlir::Value scanRefsOp = rewriter.create<mlir::subop::ScanRefsOp>(op->getLoc(), scanOp.getState(), refDef);
      rewriter.replaceOpWithNewOp<mlir::subop::GatherOp>(op, scanRefsOp, refRef, scanOp.getMapping());

      return mlir::success();
   }
};
class NormalizeSubOpPass : public mlir::PassWrapper<NormalizeSubOpPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeSubOpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-normalize"; }

   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<mlir::subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<SplitTableScan>(&getContext());
      patterns.insert<SplitGenericScan>(&getContext());

      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         assert(false && "should not happen");
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createNormalizeSubOpPass() { return std::make_unique<NormalizeSubOpPass>(); }