#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"

namespace {
//Pattern that optimizes the join order
class WrapAggrFuncPattern : public mlir::RewritePattern {
   public:
   WrapAggrFuncPattern(mlir::MLIRContext* context)
      : RewritePattern(mlir::relalg::AggrFuncOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& attributeManager = getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      mlir::relalg::AggrFuncOp aggrFuncOp = mlir::cast<mlir::relalg::AggrFuncOp>(op);
      if (mlir::isa<mlir::relalg::AggregationOp>(op->getParentOp())) {
         return mlir::success(false);
      }
      std::string scopeName = attributeManager.getUniqueScope("aggr");
      std::string attributeName = "aggrAttr";

      auto def = attributeManager.createDef(scopeName, attributeName);
      def.getColumn().type = aggrFuncOp.getType();
      auto aggrOp = rewriter.create<mlir::relalg::AggregationOp>(op->getLoc(), mlir::tuples::TupleStreamType::get(getContext()), aggrFuncOp.rel(), rewriter.getArrayAttr({}), rewriter.getArrayAttr({def}));
      auto* block = new mlir::Block;
      aggrOp.aggr_func().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto tplType = mlir::tuples::TupleType::get(getContext());
         block->addArgument(mlir::tuples::TupleStreamType::get(getContext()), op->getLoc());
         block->addArgument(tplType, op->getLoc());

         auto relArgument = block->getArgument(0);
         auto val = rewriter.create<mlir::relalg::AggrFuncOp>(op->getLoc(), aggrFuncOp.getType(), aggrFuncOp.fn(), relArgument, aggrFuncOp.attr());
         rewriter.create<mlir::tuples::ReturnOp>(op->getLoc(), mlir::ValueRange({val}));
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
      auto& attributeManager = getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      mlir::relalg::CountRowsOp aggrFuncOp = mlir::cast<mlir::relalg::CountRowsOp>(op);
      if (mlir::isa<mlir::relalg::AggregationOp>(op->getParentOp())) {
         return mlir::success(false);
      }
      std::string scopeName = attributeManager.getUniqueScope("aggr");
      std::string attributeName = "aggrAttr";

      auto def = attributeManager.createDef(scopeName, attributeName);
      def.getColumn().type = aggrFuncOp.getType();
      auto aggrOp = rewriter.create<mlir::relalg::AggregationOp>(op->getLoc(), mlir::tuples::TupleStreamType::get(getContext()), aggrFuncOp.rel(), rewriter.getArrayAttr({}), rewriter.getArrayAttr({def}));
      auto* block = new mlir::Block;
      aggrOp.aggr_func().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto tplType = mlir::tuples::TupleType::get(getContext());
         block->addArgument(mlir::tuples::TupleStreamType::get(getContext()), op->getLoc());
         block->addArgument(tplType, op->getLoc());

         auto relArgument = block->getArgument(0);
         auto val = rewriter.create<mlir::relalg::CountRowsOp>(op->getLoc(), aggrFuncOp.getType(), relArgument);
         rewriter.create<mlir::tuples::ReturnOp>(op->getLoc(), mlir::ValueRange({val}));
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

class RewriteComplexAggrFuncs : public mlir::RewritePattern {
   public:
   RewriteComplexAggrFuncs(mlir::MLIRContext* context)
      : RewritePattern(mlir::relalg::AggrFuncOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto* parentOp = op->getParentOp();
      if (!(mlir::isa<mlir::relalg::WindowOp>(parentOp) || mlir::isa<mlir::relalg::AggregationOp>(parentOp))) return mlir::failure();
      auto aggrFuncOp = mlir::cast<mlir::relalg::AggrFuncOp>(op);
      //todo: implement simplifications for stddev_samp and var_samp
      if (aggrFuncOp.fn() == mlir::relalg::AggrFunc::avg) {
         mlir::Value sum = rewriter.create<mlir::relalg::AggrFuncOp>(aggrFuncOp->getLoc(), aggrFuncOp.result().getType(), mlir::relalg::AggrFunc::sum, aggrFuncOp.rel(), aggrFuncOp.attr());
         mlir::Value count = rewriter.create<mlir::relalg::AggrFuncOp>(aggrFuncOp->getLoc(), rewriter.getI64Type(), mlir::relalg::AggrFunc::count, aggrFuncOp.rel(), aggrFuncOp.attr());
         mlir::Value casted = rewriter.create<mlir::db::CastOp>(aggrFuncOp->getLoc(), getBaseType(sum.getType()), count);
         rewriter.replaceOpWithNewOp<mlir::db::DivOp>(aggrFuncOp, sum, casted);
         return mlir::success();
      }
      return mlir::failure();
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
         patterns.insert<RewriteComplexAggrFuncs>(&getContext());

         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
      //todo: move other operators out of aggregation (map -> no problem), (projection -> problematic, using join)
      auto& attrManager = getOperation().getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      getOperation()
         .walk([&](mlir::relalg::AggregationOp aggregationOp) {
            auto scope = attrManager.getUniqueScope("aggr_rw");
            auto computedCols = aggregationOp.computed_cols();
            std::vector<mlir::Value> computedValues;
            std::vector<mlir::Attribute> computedColsAfter;
            llvm::DenseMap<mlir::Value, mlir::tuples::ColumnRefAttr> aggrMapping;
            std::vector<mlir::Attribute> colsForMap;
            std::vector<mlir::Value> valsForMap;
            auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(aggregationOp.aggr_func().front().getTerminator());
            for (size_t i = 0; i < returnOp->getNumOperands(); i++) {
               auto returnValue = returnOp.getOperand(i);
               auto isDirectAggregate = mlir::isa_and_nonnull<mlir::relalg::AggrFuncOp, mlir::relalg::CountRowsOp>(returnValue.getDefiningOp());
               if (isDirectAggregate) {
                  computedValues.push_back(returnValue);
                  computedColsAfter.push_back(computedCols[i]);
               } else {
                  colsForMap.push_back(computedCols[i]);
                  valsForMap.push_back(returnValue);
               }
            }
            size_t id = 0;
            for (auto& op : aggregationOp.aggr_func().front()) {
               if (mlir::isa<mlir::relalg::AggrFuncOp, mlir::relalg::CountRowsOp>(&op)) {
                  bool otherUser = false;
                  for (auto* user : op.getUsers()) {
                     otherUser |= !mlir::isa<mlir::tuples::ReturnOp>(user);
                  }
                  if (otherUser) {
                     auto attr = attrManager.createDef(scope, "rw" + std::to_string(id++));
                     attr.getColumn().type = op.getResult(0).getType();
                     computedValues.push_back(op.getResult(0));
                     computedColsAfter.push_back(attr);
                     aggrMapping.insert({op.getResult(0), attrManager.createRef(&attr.getColumn())});
                  }
               }
            }
            mlir::OpBuilder builder(aggregationOp);
            aggregationOp.computed_colsAttr(builder.getArrayAttr(computedColsAfter));
            returnOp->setOperands(computedValues);

            if (!colsForMap.empty()) {
               auto* block = new mlir::Block;
               builder.setInsertionPointAfter(aggregationOp);
               mlir::BlockAndValueMapping mapping;
               auto loc = aggregationOp->getLoc();
               auto newmap = builder.create<mlir::relalg::MapOp>(aggregationOp->getLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), aggregationOp, builder.getArrayAttr(colsForMap));
               newmap.predicate().push_back(block);
               auto tuple = newmap.predicate().addArgument(mlir::tuples::TupleType::get(builder.getContext()), loc);
               builder.setInsertionPointToStart(&newmap.predicate().front());
               std::vector<mlir::Operation*> getOps;
               for (auto [v, c] : aggrMapping) {
                  auto newVal = builder.create<mlir::tuples::GetColumnOp>(loc, v.getType(), c, tuple);
                  mapping.map(v, newVal);
                  getOps.push_back(newVal);
               }
               for (auto v : valsForMap) {
                  mlir::relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), &newmap.getLambdaBlock(), mapping);
               }
               for (auto* op : getOps) {
                  op->moveBefore(block, block->begin());
               }
               std::vector<mlir::Value> returnValues;
               for (auto v : valsForMap) {
                  returnValues.push_back(mapping.lookup(v));
               }
               builder.create<mlir::tuples::ReturnOp>(loc, returnValues);
               aggregationOp.result().replaceAllUsesExcept(newmap.result(), newmap);
            }
         });
      //handle distinct ones
      getOperation()
         .walk([&](mlir::relalg::AggregationOp aggregationOp) {
            mlir::Value arg = aggregationOp.aggr_func().front().getArgument(0);
            std::vector<mlir::Operation*> users(arg.getUsers().begin(), arg.getUsers().end());
            if (auto projectionOp = mlir::dyn_cast_or_null<mlir::relalg::ProjectionOp>(users[0])) {
               if (projectionOp.set_semantic() == mlir::relalg::SetSemantic::distinct) {
                  if (users.size() == 1) {
                     mlir::OpBuilder builder(aggregationOp);
                     auto cols = mlir::relalg::ColumnSet::fromArrayAttr(aggregationOp.group_by_cols());
                     cols.insert(mlir::relalg::ColumnSet::fromArrayAttr(projectionOp.cols()));
                     auto newProj = builder.create<mlir::relalg::ProjectionOp>(projectionOp->getLoc(), mlir::tuples::TupleStreamType::get(&getContext()), mlir::relalg::SetSemantic::distinct, aggregationOp.rel(), cols.asRefArrayAttr(&getContext()));
                     aggregationOp.setOperand(newProj);
                     projectionOp.replaceAllUsesWith(arg);
                     projectionOp->remove();
                     projectionOp->dropAllUses();
                     projectionOp->dropAllReferences();
                     projectionOp->destroy();
                  } else {
                     assert(false && "should not happen");
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