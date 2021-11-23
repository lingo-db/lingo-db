#include "mlir/Conversion/RelAlgToDB/HashJoinUtils.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/IR/BlockAndValueMapping.h>

class NLInnerJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::InnerJoinOp joinOp;

   public:
   NLInnerJoinLowering(mlir::relalg::InnerJoinOp innerJoinOp) : mlir::relalg::ProducerConsumerNode({innerJoinOp.left(), innerJoinOp.right()}), joinOp(innerJoinOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(this->children[1]->getAvailableAttributes());
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         children[1]->produce(context, builder);
      } else if (child == this->children[1].get()) {
         mlir::relalg::InnerJoinOp clonedInnerJoinOp = mlir::dyn_cast<mlir::relalg::InnerJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedInnerJoinOp.predicate().getBlocks().front();
         auto* terminator = block->getTerminator();

         mergeRelatinalBlock(builder.getInsertionBlock(), block, context, scope);

         auto builderValuesBefore = getRequiredBuilderValues(context);
         auto ifOp = builder.create<mlir::db::IfOp>(
            joinOp->getLoc(), getRequiredBuilderTypes(context), mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0], [&](mlir::OpBuilder& builder1, mlir::Location) {
               consumer->consume(this, builder1, context);
               builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
            requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
         setRequiredBuilderValues(context, ifOp.getResults());
         terminator->erase();
         clonedInnerJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLInnerJoinLowering() {}
};

class HashInnerJoinLowering : public mlir::relalg::HJNode<mlir::relalg::InnerJoinOp> {
   public:
   HashInnerJoinLowering(mlir::relalg::InnerJoinOp innerJoinOp) : mlir::relalg::HJNode<mlir::relalg::InnerJoinOp>(innerJoinOp, innerJoinOp.left(), innerJoinOp.right()) {
   }

   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(this->children[1]->getAvailableAttributes());
   }
   virtual void handleLookup(mlir::Value matched, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
            consumer->consume(this, builder1, context);
            builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
      setRequiredBuilderValues(context, ifOp.getResults());
   }

   virtual ~HashInnerJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredInnerJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::InnerJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashInnerJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLInnerJoinLowering>(joinOp);
});