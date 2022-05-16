#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include <iostream>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
class ReduceAggrKeys : public mlir::RewritePattern {
   public:
   ReduceAggrKeys(mlir::MLIRContext* context)
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
         auto *terminator = aggr.aggr_func().begin()->getTerminator();
         rewriter.setInsertionPoint(terminator);
         auto returnArgs = mlir::cast<mlir::relalg::ReturnOp>(terminator)->getOperands();
         std::vector<mlir::Value> values(returnArgs.begin(), returnArgs.end());
         for (auto* x : toMap) {
            auto *newCol = colManager.get(scope, colManager.getName(x).second).get();
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
class PropagateConstraints : public mlir::PassWrapper<PropagateConstraints, mlir::OperationPass<mlir::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-propagate-constrains"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<ReduceAggrKeys>(&getContext());

         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createPropagateConstraintsPass() { return std::make_unique<PropagateConstraints>(); }
} // end namespace relalg
} // end namespace mlir