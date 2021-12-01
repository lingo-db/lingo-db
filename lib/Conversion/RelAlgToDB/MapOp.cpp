#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MapTranslator : public mlir::relalg::Translator {
   public:
   MapTranslator(mlir::relalg::MapOp mapOp) : mlir::relalg::Translator(mapOp) {
   }

   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      mergeRelationalBlock(
         builder.getInsertionBlock(), op, [](auto x) { return &x->getRegion(0).front(); }, context, scope);
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~MapTranslator() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredMapOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::MapOp mapOp) {
  return std::make_unique<MapTranslator>(mapOp);
});