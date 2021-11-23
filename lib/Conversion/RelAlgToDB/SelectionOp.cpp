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
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
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

      mergeRelatinalBlock(builder.getInsertionBlock(), block, context, scope);
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         selectionOp->getLoc(), getRequiredBuilderTypes(context), mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0], [&](mlir::OpBuilder& builder1, mlir::Location) {
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(selectionOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location loc) { builder2.create<mlir::db::YieldOp>(loc, builderValuesBefore); });
      setRequiredBuilderValues(context, ifOp.getResults());
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