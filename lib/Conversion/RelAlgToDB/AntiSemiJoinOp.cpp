#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/HashJoinUtils.h>
#include <mlir/IR/BlockAndValueMapping.h>
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
         mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
         mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);
         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound);
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
         builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, matched);
         terminator->erase();
         clonedAntiSemiJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLAntiSemiJoinLowering() {}
};

class HashAntiSemiJoinLowering : public mlir::relalg::HJNode<mlir::relalg::AntiSemiJoinOp> {
   mlir::Value matchFoundFlag;

   public:
   HashAntiSemiJoinLowering(mlir::relalg::AntiSemiJoinOp innerJoinOp) : mlir::relalg::HJNode<mlir::relalg::AntiSemiJoinOp>(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, matched);
   }
   mlir::Value getFlag() override {
      return matchFoundFlag;
   }
   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);

      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound);
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
   }
   virtual ~HashAntiSemiJoinLowering() {}
};
class MHashAntiSemiJoinLowering : public mlir::relalg::MarkableHJNode<mlir::relalg::AntiSemiJoinOp> {
   public:
   MHashAntiSemiJoinLowering(mlir::relalg::AntiSemiJoinOp innerJoinOp) : mlir::relalg::MarkableHJNode<mlir::relalg::AntiSemiJoinOp>(innerJoinOp, innerJoinOp.left(), innerJoinOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matched);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::relalg::ProducerConsumerBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      auto const1 = builder1.create<mlir::ConstantOp>(builder1.getUnknownLoc(), builder1.getIntegerType(8), builder1.getI8IntegerAttr(1));
      builder1.create<mlir::AtomicRMWOp>(builder1.getUnknownLoc(), builder1.getIntegerType(8), mlir::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
   }
   virtual void after(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto zero = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::CmpIOp>(builder.getUnknownLoc(), mlir::CmpIPredicate::eq, marker, zero);
      auto isZeroDB = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), isZero);
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), isZeroDB);
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
   }

   virtual ~MHashAntiSemiJoinLowering() {}
};
bool mlir::relalg::ProducerConsumerNodeRegistry::registeredAntiSemiJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::AntiSemiJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashAntiSemiJoinLowering>(joinOp);
         }
         if (impl.getValue() == "markhash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<MHashAntiSemiJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLAntiSemiJoinLowering>(joinOp);
});