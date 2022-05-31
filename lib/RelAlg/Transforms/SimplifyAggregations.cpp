#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
//Pattern that optimizes the join order
class WrapAggrFuncPattern : public mlir::RewritePattern {
   public:
   WrapAggrFuncPattern(mlir::MLIRContext* context)
      : RewritePattern(mlir::relalg::AggrFuncOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& attributeManager = getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

      mlir::relalg::AggrFuncOp aggrFuncOp = mlir::cast<mlir::relalg::AggrFuncOp>(op);
      if (mlir::isa<mlir::relalg::AggregationOp>(op->getParentOp())) {
         return mlir::success(false);
      }
      std::string scopeName = attributeManager.getUniqueScope("aggr");
      std::string attributeName = "aggrAttr";

      auto def = attributeManager.createDef(scopeName, attributeName);
      def.getColumn().type = aggrFuncOp.getType();
      auto aggrOp = rewriter.create<mlir::relalg::AggregationOp>(op->getLoc(), mlir::relalg::TupleStreamType::get(getContext()), aggrFuncOp.rel(), rewriter.getArrayAttr({}), rewriter.getArrayAttr({def}));
      auto* block = new mlir::Block;
      aggrOp.aggr_func().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto tplType = mlir::relalg::TupleType::get(getContext());
         block->addArgument(mlir::relalg::TupleStreamType::get(getContext()), op->getLoc());
         block->addArgument(tplType, op->getLoc());

         auto relArgument = block->getArgument(0);
         auto val = rewriter.create<mlir::relalg::AggrFuncOp>(op->getLoc(), aggrFuncOp.getType(), aggrFuncOp.fn(), relArgument, aggrFuncOp.attr());
         rewriter.create<mlir::relalg::ReturnOp>(op->getLoc(), mlir::ValueRange({val}));
      }
      auto nullableType = aggrFuncOp.getType().dyn_cast_or_null<mlir::db::NullableType>();
      mlir::Value getScalarOp = rewriter.replaceOpWithNewOp<mlir::relalg::GetScalarOp>(op, nullableType, attributeManager.createRef(&def.getColumn()), aggrOp.asRelation());
      mlir::Value res = getScalarOp;
      if (!nullableType) {
         res = rewriter.create<mlir::db::NullableGetVal>(op->getLoc(), aggrFuncOp.getType(), getScalarOp);
      }
      rewriter.replaceOp(op, res);
      return mlir::success(true);
   }
};
class WrapCountRowsPattern : public mlir::RewritePattern {
   public:
   WrapCountRowsPattern(mlir::MLIRContext* context)
      : RewritePattern(mlir::relalg::CountRowsOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& attributeManager = getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

      mlir::relalg::CountRowsOp aggrFuncOp = mlir::cast<mlir::relalg::CountRowsOp>(op);
      if (mlir::isa<mlir::relalg::AggregationOp>(op->getParentOp())) {
         return mlir::success(false);
      }
      std::string scopeName = attributeManager.getUniqueScope("aggr");
      std::string attributeName = "aggrAttr";

      auto def = attributeManager.createDef(scopeName, attributeName);
      def.getColumn().type = aggrFuncOp.getType();
      auto aggrOp = rewriter.create<mlir::relalg::AggregationOp>(op->getLoc(), mlir::relalg::TupleStreamType::get(getContext()), aggrFuncOp.rel(), rewriter.getArrayAttr({}), rewriter.getArrayAttr({def}));
      auto* block = new mlir::Block;
      aggrOp.aggr_func().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto tplType = mlir::relalg::TupleType::get(getContext());
         block->addArgument(mlir::relalg::TupleStreamType::get(getContext()), op->getLoc());
         block->addArgument(tplType, op->getLoc());

         auto relArgument = block->getArgument(0);
         auto val = rewriter.create<mlir::relalg::CountRowsOp>(op->getLoc(), aggrFuncOp.getType(), relArgument);
         rewriter.create<mlir::relalg::ReturnOp>(op->getLoc(), mlir::ValueRange({val}));
      }
      mlir::Type nullableType = aggrFuncOp.getType();
      if (!nullableType.isa<mlir::db::NullableType>()) {
         nullableType = mlir::db::NullableType::get(rewriter.getContext(), nullableType);
      }
      mlir::Value getScalarOp = rewriter.create<mlir::relalg::GetScalarOp>(op->getLoc(), nullableType, attributeManager.createRef(&def.getColumn()), aggrOp.asRelation());
      mlir::Value res = rewriter.create<mlir::db::AsNullableOp>(op->getLoc(), aggrFuncOp.getType(), getScalarOp);
      rewriter.replaceOp(op, res);
      return mlir::success(true);
   }
};
class SimplifyAggregations : public mlir::PassWrapper<SimplifyAggregations, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-simplify-aggrs"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<WrapAggrFuncPattern>(&getContext());
         patterns.insert<WrapCountRowsPattern>(&getContext());

         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }

      //handle distinct ones
      getOperation()
         .walk([&](mlir::relalg::AggregationOp aggregationOp) {
            mlir::Value arg = aggregationOp.aggr_func().front().getArgument(0);
            std::vector<mlir::Operation*> users(arg.getUsers().begin(), arg.getUsers().end());
            if (users.size() == 1) {
               if (auto projectionOp = mlir::dyn_cast_or_null<mlir::relalg::ProjectionOp>(users[0])) {
                  if (projectionOp.set_semantic() == mlir::relalg::SetSemantic::distinct) {
                     mlir::OpBuilder builder(aggregationOp);
                     auto cols = mlir::relalg::ColumnSet::fromArrayAttr(aggregationOp.group_by_cols());
                     cols.insert(mlir::relalg::ColumnSet::fromArrayAttr(projectionOp.cols()));
                     auto newProj = builder.create<mlir::relalg::ProjectionOp>(projectionOp->getLoc(), mlir::relalg::TupleStreamType::get(&getContext()), mlir::relalg::SetSemantic::distinct, aggregationOp.rel(), cols.asRefArrayAttr(&getContext()));
                     aggregationOp.setOperand(newProj);
                     projectionOp.replaceAllUsesWith(arg);
                     projectionOp->remove();
                     projectionOp->dropAllUses();
                     projectionOp->dropAllReferences();
                     projectionOp->destroy();
                  }
               }
            }
         });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createSimplifyAggregationsPass() { return std::make_unique<SimplifyAggregations>(); }
} // end namespace relalg
} // end namespace mlir