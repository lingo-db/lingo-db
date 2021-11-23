#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class SelectionLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::SelectionOp selectionOp;

   public:
   SelectionLowering(mlir::relalg::SelectionOp selectionOp) : mlir::relalg::ProducerConsumerNode(selectionOp.rel()), selectionOp(selectionOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(selectionOp.getUsedAttributes());
      propagateInfo();
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      mlir::relalg::SelectionOp clonedSelectionOp = mlir::dyn_cast<mlir::relalg::SelectionOp>(selectionOp->clone());
      mlir::Block* block = &clonedSelectionOp.predicate().getBlocks().front();
      auto* terminator = block->getTerminator();

      mergeRelatinalBlock(builder.getInsertionBlock(),block, context, scope);

      auto ifOp = builder.create<mlir::db::IfOp>(selectionOp->getLoc(), getRequiredBuilderTypes(context), mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0]);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::OpBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::OpBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(selectionOp->getLoc(), getRequiredBuilderValues(context));
      }
      consumer->consume(this, builder1, context);
      builder1.create<mlir::db::YieldOp>(selectionOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
      terminator->erase();
      clonedSelectionOp->destroy();
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~SelectionLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredSelectionOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::SelectionOp selectionOp) {
  return std::make_unique<SelectionLowering>(selectionOp);
});