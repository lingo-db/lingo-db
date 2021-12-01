#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class SelectionTranslator : public mlir::relalg::Translator {
   mlir::relalg::SelectionOp selectionOp;

   public:
   SelectionTranslator(mlir::relalg::SelectionOp selectionOp) : mlir::relalg::Translator(selectionOp), selectionOp(selectionOp) {
   }

   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();

      mlir::Value matched = mergeRelationalBlock(
         builder.getInsertionBlock(), selectionOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         selectionOp->getLoc(), getRequiredBuilderTypes(context), matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(selectionOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location loc) { builder2.create<mlir::db::YieldOp>(loc, builderValuesBefore); });
      setRequiredBuilderValues(context, ifOp.getResults());
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~SelectionTranslator() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredSelectionOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::SelectionOp selectionOp) {
   return std::make_unique<SelectionTranslator>(selectionOp);
});