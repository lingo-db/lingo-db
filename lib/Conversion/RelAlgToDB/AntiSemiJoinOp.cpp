#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class NLAntiSemiJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::AntiSemiJoinOp joinOp;
   mlir::Value matchFoundFlag;

   public:
   NLAntiSemiJoinLowering(mlir::relalg::AntiSemiJoinOp innerJoinOp) : mlir::relalg::ProducerConsumerNode({innerJoinOp.left(), innerJoinOp.right()}), joinOp(innerJoinOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
         children[1]->setFlag(matchFoundFlag);
         children[1]->produce(context, builder);
         mlir::Value matchFound=builder.create<mlir::db::GetFlag>(joinOp->getLoc(),mlir::db::BoolType::get(builder.getContext()),matchFoundFlag);
         mlir::Value noMatchFound=builder.create<mlir::db::NotOp>(joinOp->getLoc(),mlir::db::BoolType::get(builder.getContext()),matchFound);
         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context),noMatchFound);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::relalg::ProducerConsumerBuilder builder2(ifOp.elseRegion());
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

         size_t i = 0;
         for (auto b : requiredBuilders) {
            context.builders[b] = ifOp.getResult(i++);
         }
      } else if (child == this->children[1].get()) {
         mlir::relalg::AntiSemiJoinOp clonedAntiSemiJoinOp = mlir::dyn_cast<mlir::relalg::AntiSemiJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedAntiSemiJoinOp.predicate().getBlocks().front();
         auto* terminator = block->getTerminator();

         builder.mergeRelatinalBlock(block, context, scope);
         mlir::Value matched = mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0];
         builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag,matched);
         terminator->erase();
         clonedAntiSemiJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLAntiSemiJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredAntiSemiJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::AntiSemiJoinOp joinOp) {
   return std::make_unique<NLAntiSemiJoinLowering>(joinOp);
});