#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

#include <mlir/Conversion/RelAlgToDB/HashJoinUtils.h>
#include <mlir/IR/BlockAndValueMapping.h>

class NLSemiJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::SemiJoinOp joinOp;
   mlir::Value matchFoundFlag;

   public:
   NLSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::ProducerConsumerNode({innerJoinOp.left(), innerJoinOp.right()}), joinOp(innerJoinOp) {
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
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
         children[1]->setFlag(matchFoundFlag);
         children[1]->produce(context, builder);
         mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matchFound);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::OpBuilder builder1(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::OpBuilder builder2(ifOp.elseRegion());
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

         size_t i = 0;
         for (auto b : requiredBuilders) {
            context.builders[b] = ifOp.getResult(i++);
         }
      } else if (child == this->children[1].get()) {
         mlir::relalg::SemiJoinOp clonedSemiJoinOp = mlir::dyn_cast<mlir::relalg::SemiJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedSemiJoinOp.predicate().getBlocks().front();
         auto* terminator = block->getTerminator();

         mergeRelatinalBlock(builder.getInsertionBlock(), block, context, scope);
         mlir::Value matched = mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0];
         builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, matched);
         terminator->erase();
         clonedSemiJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLSemiJoinLowering() {}
};

class HashSemiJoinLowering : public mlir::relalg::HJNode<mlir::relalg::SemiJoinOp> {
   mlir::Value matchFoundFlag;

   public:
   HashSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::HJNode<mlir::relalg::SemiJoinOp>(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, matched);
   }
   mlir::Value getFlag() override {
      return matchFoundFlag;
   }
   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matchFound);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::OpBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::OpBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      consumer->consume(this, builder1, context);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
   }
   virtual ~HashSemiJoinLowering() {}
};
class MHashSemiJoinLowering : public mlir::relalg::MarkableHJNode<mlir::relalg::SemiJoinOp> {
   public:
   MHashSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::MarkableHJNode<mlir::relalg::SemiJoinOp>(innerJoinOp, innerJoinOp.left(), innerJoinOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matched);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::OpBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::OpBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      auto const1 = builder1.create<mlir::arith::ConstantOp>(builder1.getUnknownLoc(), builder1.getIntegerType(8), builder1.getI8IntegerAttr(1));
      auto markerBefore = builder1.create<mlir::AtomicRMWOp>(builder1.getUnknownLoc(), builder1.getIntegerType(8), mlir::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
      {
         auto zero = builder1.create<mlir::arith::ConstantOp>(builder1.getUnknownLoc(), markerBefore.getType(), builder1.getIntegerAttr(markerBefore.getType(), 0));
         auto isZero = builder1.create<mlir::arith::CmpIOp>(builder1.getUnknownLoc(), mlir::arith::CmpIPredicate::eq, markerBefore, zero);
         auto isZeroDB = builder1.create<mlir::db::TypeCastOp>(builder1.getUnknownLoc(), mlir::db::BoolType::get(builder1.getContext()), isZero);
         auto ifOp = builder1.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), isZeroDB);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::OpBuilder builder10(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::OpBuilder builder20(ifOp.elseRegion());
            builder20.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         consumer->consume(this, builder10, context);
         builder10.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

         size_t i = 0;
         for (auto b : requiredBuilders) {
            context.builders[b] = ifOp.getResult(i++);
         }
      }
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
   }

   virtual ~MHashSemiJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredSemiJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::SemiJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashSemiJoinLowering>(joinOp);
         }
         if (impl.getValue() == "markhash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<MHashSemiJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLSemiJoinLowering>(joinOp);
});