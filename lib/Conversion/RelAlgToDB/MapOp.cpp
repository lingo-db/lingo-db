#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MapLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::MapOp mapOp;

   public:
   MapLowering(mlir::relalg::MapOp mapOp) : mlir::relalg::ProducerConsumerNode(mapOp.rel()), mapOp(mapOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(mapOp.getUsedAttributes());
      propagateInfo();
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(mapOp.getCreatedAttributes());
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      mlir::relalg::MapOp clonedSelectionOp = mlir::dyn_cast<mlir::relalg::MapOp>(mapOp->clone());
      mlir::Block* block = &clonedSelectionOp.predicate().getBlocks().front();
      auto* terminator = block->getTerminator();

      mergeRelatinalBlock(builder.getInsertionBlock(),block, context, scope);
      consumer->consume(this, builder, context);
      terminator->erase();
      clonedSelectionOp->destroy();
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~MapLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredMapOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::MapOp mapOp) {
  return std::make_unique<MapLowering>(mapOp);
});