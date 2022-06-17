#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <unordered_set>

#include "llvm/ADT/EquivalenceClasses.h"

namespace {
using ReplaceFnT = std::function<mlir::relalg::ColumnRefAttr(mlir::relalg::ColumnRefAttr)>;
mlir::Attribute updateAttribute(mlir::Attribute attr, ReplaceFnT replaceFn) {
   if (auto colRefAttr = attr.dyn_cast<mlir::relalg::ColumnRefAttr>()) {
      return replaceFn(colRefAttr);
   }
   if (auto colDefAttr = attr.dyn_cast<mlir::relalg::ColumnDefAttr>()) {
      return mlir::relalg::ColumnDefAttr::get(attr.getContext(), colDefAttr.getName(), colDefAttr.getColumnPtr(), updateAttribute(colDefAttr.getFromExisting(), replaceFn));
   }
   if (auto sortSpec = attr.dyn_cast<mlir::relalg::SortSpecificationAttr>()) {
      return mlir::relalg::SortSpecificationAttr::get(attr.getContext(), replaceFn(sortSpec.getAttr()), sortSpec.getSortSpec());
   }
   if (auto arrayAttr = attr.dyn_cast<mlir::ArrayAttr>()) {
      std::vector<mlir::Attribute> attributes;
      for (auto elem : arrayAttr) {
         attributes.push_back(updateAttribute(elem, replaceFn));
      }
      return mlir::ArrayAttr::get(attr.getContext(), attributes);
   }
   return attr;
}
void replaceUsages(mlir::Operation* op, ReplaceFnT replaceFn) {
   for (auto attr : op->getAttrs()) {
      op->setAttr(attr.getName(), updateAttribute(attr.getValue(), replaceFn));
   }
}

void replaceUsagesAfter(mlir::Operation* op, ReplaceFnT replaceFn) {
   mlir::Operation* current = op->getNextNode();
   while (current) {
      current->walk([&replaceFn](mlir::Operation* op) {
         replaceUsages(op, replaceFn);
      });
      current = current->getNextNode();
   }
}
class ReduceAggrKeyPattern : public mlir::RewritePattern {
   public:
   ReduceAggrKeyPattern(mlir::MLIRContext* context)
      : RewritePattern(mlir::relalg::AggregationOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto aggr = mlir::cast<mlir::relalg::AggregationOp>(op);
      if (auto child = mlir::dyn_cast_or_null<Operator>(aggr.rel().getDefiningOp())) {
         auto fds = child.getFDs();
         auto keys = mlir::relalg::ColumnSet::fromArrayAttr(aggr.group_by_cols());
         auto reducedKeys = fds.reduce(keys);
         if (reducedKeys.size() == keys.size()) {
            return mlir::failure();
         }
         auto toMap = keys;
         toMap.remove(reducedKeys);
         aggr.group_by_colsAttr(reducedKeys.asRefArrayAttr(aggr->getContext()));
         auto& colManager = getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
         auto scope = colManager.getUniqueScope("aggr");
         std::unordered_map<const mlir::relalg::Column*, const mlir::relalg::Column*> mapping;
         std::vector<mlir::Attribute> computedCols(aggr.computed_cols().begin(), aggr.computed_cols().end());
         auto* terminator = aggr.aggr_func().begin()->getTerminator();
         rewriter.setInsertionPoint(terminator);
         auto returnArgs = mlir::cast<mlir::relalg::ReturnOp>(terminator)->getOperands();
         std::vector<mlir::Value> values(returnArgs.begin(), returnArgs.end());
         for (auto* x : toMap) {
            auto* newCol = colManager.get(scope, colManager.getName(x).second).get();
            newCol->type = x->type;
            mapping.insert({x, newCol});
            computedCols.push_back(colManager.createDef(newCol));
            values.push_back(rewriter.create<mlir::relalg::AggrFuncOp>(aggr->getLoc(), x->type, mlir::relalg::AggrFunc::any, aggr.aggr_func().getArgument(0), colManager.createRef(x)));
         }
         rewriter.create<mlir::relalg::ReturnOp>(aggr->getLoc(), values);
         rewriter.eraseOp(terminator);
         aggr.computed_colsAttr(mlir::ArrayAttr::get(aggr.getContext(), computedCols));
         replaceUsagesAfter(aggr.getOperation(), [&](mlir::relalg::ColumnRefAttr attr) {
            if (mapping.count(&attr.getColumn())) {
               return colManager.createRef(mapping.at(&attr.getColumn()));
            }
            return attr;
         });
         std::cout << "before: " << keys.size() << " after: " << reducedKeys.size() << std::endl;
         return mlir::success();
      }
      return mlir::failure();
   }
};

class ReduceAggrKeys : public mlir::PassWrapper<ReduceAggrKeys, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-reduce-aggr-keys"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<ReduceAggrKeyPattern>(&getContext());

         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
static std::optional<std::pair<const mlir::relalg::Column*, const mlir::relalg::Column*>> analyzePredicate(PredicateOperator selection) {
   auto returnOp = mlir::cast<mlir::relalg::ReturnOp>(selection.getPredicateBlock().getTerminator());
   if (returnOp.results().empty()) return {};
   mlir::Value v = returnOp.results()[0];
   if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(v.getDefiningOp())) {
      if (!cmpOp.isEqualityPred()) return {};
      if (auto leftColref = mlir::dyn_cast_or_null<mlir::relalg::GetColumnOp>(cmpOp.left().getDefiningOp())) {
         if (auto rightColref = mlir::dyn_cast_or_null<mlir::relalg::GetColumnOp>(cmpOp.right().getDefiningOp())) {
            return std::make_pair<const mlir::relalg::Column*, const mlir::relalg::Column*>(&leftColref.attr().getColumn(), &rightColref.attr().getColumn());
         }
      }
   }
   return {};
}

class ExpandTransitiveEqualities : public mlir::PassWrapper<ExpandTransitiveEqualities, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-expand-transitive-eq"; }

   void merge(llvm::EquivalenceClasses<const mlir::relalg::Column*>& mergeInto, const llvm::EquivalenceClasses<const mlir::relalg::Column*>& mergeFrom, std::vector<std::pair<const mlir::relalg::Column*, const mlir::relalg::Column*>>& additionalConstrains) {
      llvm::EquivalenceClasses<const mlir::relalg::Column*> before = mergeInto;
      for (auto& x : mergeFrom) {
         mergeInto.unionSets(x.getData(), mergeFrom.getLeaderValue(x.getData()));
      }
      for (auto& x : mergeInto) {
         const mlir::relalg::Column* xData = x.getData();
         for (auto& y : mergeInto) {
            const mlir::relalg::Column* yData = y.getData();
            if (yData < xData && mergeInto.isEquivalent(xData, yData) && !before.isEquivalent(xData, yData) && !mergeFrom.isEquivalent(xData, yData)) {
               auto& colManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
               auto xName = colManager.getName(xData);
               auto yName = colManager.getName(yData);
               additionalConstrains.push_back(std::make_pair(xData, yData));
               //std::cout << xName.first << "::" << xName.second << "=" << yName.first << "::" << yName.second << "\n";
               //std::cout << yData << " < " << xData << std::endl;
            }
         }
      }
   }

   public:
   void runOnOperation() override {
      std::unordered_map<mlir::Operation*, llvm::EquivalenceClasses<const mlir::relalg::Column*>> equalities;

      getOperation().walk([&](Operator op) {
         llvm::EquivalenceClasses<const mlir::relalg::Column*> localEqualities;
         std::vector<std::pair<const mlir::relalg::Column*, const mlir::relalg::Column*>> additionalPredicates;
         if (auto unary = mlir::dyn_cast<UnaryOperator>(op.getOperation())) {
            auto previous = equalities[unary.child()];
            merge(localEqualities, previous, additionalPredicates);
         }

         if (auto cp = mlir::dyn_cast<mlir::relalg::CrossProductOp>(op.getOperation())) {
            merge(localEqualities, equalities[cp.left().getDefiningOp()], additionalPredicates);
            merge(localEqualities, equalities[cp.right().getDefiningOp()], additionalPredicates);
         }
         if (auto ij = mlir::dyn_cast<mlir::relalg::InnerJoinOp>(op.getOperation())) {
            merge(localEqualities, equalities[ij.left().getDefiningOp()], additionalPredicates);
            merge(localEqualities, equalities[ij.right().getDefiningOp()], additionalPredicates);
         }

         if (auto sel = mlir::dyn_cast<mlir::relalg::SelectionOp>(op.getOperation())) {
            auto res = analyzePredicate(sel);
            if (res) {
               llvm::EquivalenceClasses<const mlir::relalg::Column*> selPred;
               selPred.unionSets(res.value().first, res.value().second);
               merge(localEqualities, selPred, additionalPredicates);
            }
         }
         mlir::Value current = op.asRelation();
         mlir::OpBuilder builder(&getContext());
         builder.setInsertionPointAfter(op.getOperation());
         auto& colManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
         auto availableColumns = op.getAvailableColumns();
         for (auto pred : additionalPredicates) {
            auto loc = builder.getUnknownLoc();
            auto* block = new mlir::Block;
            mlir::OpBuilder predBuilder(&getContext());
            block->addArgument(mlir::relalg::TupleType::get(&getContext()), loc);

            predBuilder.setInsertionPointToStart(block);
            if (availableColumns.contains(pred.first) && availableColumns.contains(pred.second)) {
               mlir::Value left = predBuilder.create<mlir::relalg::GetColumnOp>(loc, pred.first->type, colManager.createRef(pred.first), block->getArgument(0));
               mlir::Value right = predBuilder.create<mlir::relalg::GetColumnOp>(loc, pred.second->type, colManager.createRef(pred.second), block->getArgument(0));
               mlir::Value compared = predBuilder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, left, right);
               predBuilder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), compared);

               auto sel = builder.create<mlir::relalg::SelectionOp>(loc, mlir::relalg::TupleStreamType::get(builder.getContext()), current);
               sel.predicate().push_back(block);
               current.replaceAllUsesExcept(sel.asRelation(), sel.getOperation());
               current = sel.asRelation();
            }
         }

         equalities[op] = localEqualities;
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createReduceGroupByKeysPass() { return std::make_unique<ReduceAggrKeys>(); }
std::unique_ptr<Pass> createExpandTransitiveEqualities() { return std::make_unique<ExpandTransitiveEqualities>(); }

} // end namespace relalg
} // end namespace mlir