#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class CrossProductLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::CrossProductOp crossProductOp;

   public:
   CrossProductLowering(mlir::relalg::CrossProductOp crossProductOp) : mlir::relalg::ProducerConsumerNode(mlir::ValueRange({crossProductOp.left(), crossProductOp.right()})), crossProductOp(crossProductOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(this->children[1]->getAvailableAttributes());
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      if (child == this->children[0].get()) {
         children[1]->produce(context, builder);
      } else if (child == this->children[1].get()) {
         consumer->consume(this, builder, context);
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~CrossProductLowering() {}
};
bool mlir::relalg::ProducerConsumerNodeRegistry::registeredCrossProductOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::CrossProductOp crossProductOp) {
  return std::make_unique<CrossProductLowering>(crossProductOp);
});