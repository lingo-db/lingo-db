#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MapLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::MapOp mapOp;

   public:
   MapLowering(mlir::relalg::MapOp mapOp) : mlir::relalg::ProducerConsumerNode(mapOp), mapOp(mapOp) {
   }

   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }

   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      mergeRelationalBlock(
         builder.getInsertionBlock(), mapOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope);
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~MapLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredMapOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::MapOp mapOp) {
  return std::make_unique<MapLowering>(mapOp);
});