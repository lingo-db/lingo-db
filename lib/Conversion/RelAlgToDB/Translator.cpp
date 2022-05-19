#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include <mlir/Conversion/RelAlgToDB/HashJoinTranslator.h>
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>

using namespace mlir::relalg;
std::vector<mlir::Value> mlir::relalg::Translator::mergeRelationalBlock(mlir::Block* dest, mlir::Operation* op, mlir::function_ref<mlir::Block*(mlir::Operation*)> getBlockFn, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope) {
   // Splice the operations of the 'source' block into the 'dest' block and erase
   // it.
   llvm::iplist<mlir::Operation> translated;
   std::vector<mlir::Operation*> toErase;
   auto* cloned = op->clone();
   mlir::Block* source = getBlockFn(cloned);
   auto* terminator = source->getTerminator();

   source->walk([&](mlir::relalg::GetColumnOp getColumnOp) {
      getColumnOp.replaceAllUsesWith(context.getValueForAttribute(&getColumnOp.attr().getColumn()));
      toErase.push_back(getColumnOp.getOperation());
   });
   /*for (auto addColumnOp : source->getOps<mlir::relalg::AddColumnOp>()) {
      context.setValueForAttribute(scope, &addColumnOp.attr().getColumn(), addColumnOp.val());
      toErase.push_back(addColumnOp.getOperation());
   }*/

   dest->getOperations().splice(dest->end(), source->getOperations());
   for (auto* op : toErase) {
      op->dropAllUses();
      op->erase();
   }
   auto returnOp = mlir::cast<mlir::relalg::ReturnOp>(terminator);
   std::vector<Value> res(returnOp.results().begin(), returnOp.results().end());
   terminator->erase();
   return res;
}

void Translator::propagateInfo() {
   for (auto& c : children) {
      auto available = c->getAvailableColumns();
      mlir::relalg::ColumnSet toPropagate = requiredAttributes.intersect(available);
      c->setInfo(this, toPropagate);
   }
}
mlir::relalg::Translator::Translator(Operator op) : op(op) {
   for (auto child : op.getChildren()) {
      children.push_back(mlir::relalg::Translator::createTranslator(child.getOperation()));
   }
}

mlir::relalg::Translator::Translator(mlir::ValueRange potentialChildren) : op() {
   for (auto child : potentialChildren) {
      if (child.getType().isa<mlir::relalg::TupleStreamType>()) {
         children.push_back(mlir::relalg::Translator::createTranslator(child.getDefiningOp()));
      }
   }
}

void Translator::setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) {
   this->consumer = consumer;
   this->requiredAttributes = requiredAttributes;
   if (op) {
      this->requiredAttributes.insert(op.getUsedColumns());
      propagateInfo();
   }
}
mlir::relalg::ColumnSet Translator::getAvailableColumns() {
   return op.getAvailableColumns();
};

std::unique_ptr<mlir::relalg::Translator> Translator::createTranslator(mlir::Operation* operation) {
   return ::llvm::TypeSwitch<mlir::Operation*, std::unique_ptr<mlir::relalg::Translator>>(operation)
      .Case<BaseTableOp>([&](auto x) { return createBaseTableTranslator(x); })
      .Case<ConstRelationOp>([&](auto x) { return createConstRelTranslator(x); })
      .Case<MaterializeOp>([&](auto x) { return createMaterializeTranslator(x); })
      .Case<SelectionOp>([&](auto x) { return createSelectionTranslator(x); })
      .Case<MapOp>([&](auto x) { return createMapTranslator(x); })
      .Case<CrossProductOp, InnerJoinOp, SemiJoinOp, AntiSemiJoinOp, OuterJoinOp, SingleJoinOp, MarkJoinOp, CollectionJoinOp>([&](mlir::Operation* op) { return createJoinTranslator(op); })
      .Case<SortOp>([&](auto x) { return createSortTranslator(x); })
      .Case<AggregationOp>([&](auto x) { return createAggregationTranslator(x); })
      .Case<RenamingOp>([&](auto x) { return createRenamingTranslator(x); })
      .Case<ProjectionOp>([&](auto x) { return createProjectionTranslator(x); })
      .Case<LimitOp>([&](auto x) { return createLimitTranslator(x); })
      .Case<TmpOp>([&](auto x) { return createTmpTranslator(x); })
      .Case<UnionOp, IntersectOp, ExceptOp>([&](auto x) { return createSetOpTranslator(x); })
      .Default([](auto x) { assert(false&&"should not happen"); return std::unique_ptr<Translator>(); });
}
