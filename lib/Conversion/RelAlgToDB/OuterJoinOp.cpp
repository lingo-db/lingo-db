#include "mlir/Conversion/RelAlgToDB/HashJoinUtils.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/IR/BlockAndValueMapping.h>

class NLOuterJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::OuterJoinOp joinOp;
   mlir::Value matchFoundFlag;

   public:
   NLOuterJoinLowering(mlir::relalg::OuterJoinOp outerJoinOp) : mlir::relalg::ProducerConsumerNode({outerJoinOp.left(), outerJoinOp.right()}), joinOp(outerJoinOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
         }
      }
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return joinOp.getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
         children[1]->produce(context, builder);
         mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
         mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);
         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::OpBuilder builder1(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::OpBuilder builder2(ifOp.elseRegion());
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         for (mlir::Attribute attr : joinOp.mapping()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if (this->requiredAttributes.contains(defAttr)) {
               auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);

               context.setValueForAttribute(scope, defAttr, nullValue);
            }
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, ifOp.getResults());
      } else if (child == this->children[1].get()) {
         mlir::relalg::OuterJoinOp clonedOuterJoinOp = mlir::dyn_cast<mlir::relalg::OuterJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedOuterJoinOp.predicate().getBlocks().front();
         auto* terminator = block->getTerminator();

         mergeRelatinalBlock(builder.getInsertionBlock(),block, context, scope);

         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0]);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::OpBuilder builder1(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::OpBuilder builder2(ifOp.elseRegion());
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         for (mlir::Attribute attr : joinOp.mapping()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if (this->requiredAttributes.contains(defAttr)) {
               auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
               auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
               auto value = context.getValueForAttribute(refAttr);
               if (refAttr->type != defAttr->type) {
                  mlir::Value tmp = builder1.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
                  value = tmp;
               }
               context.setValueForAttribute(scope, defAttr, value);
            }
         }
         consumer->consume(this, builder1, context);
         auto trueVal = builder1.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder1.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, trueVal);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

         size_t i = 0;
         for (auto b : requiredBuilders) {
            context.builders[b] = ifOp.getResult(i++);
         }
         terminator->erase();
         clonedOuterJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLOuterJoinLowering() {}
};

class HashOuterJoinLowering : public mlir::relalg::HJNode<mlir::relalg::OuterJoinOp> {
   mlir::Value matchFoundFlag;

   public:
   HashOuterJoinLowering(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::HJNode<mlir::relalg::OuterJoinOp>(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }
   void addAdditionalRequiredAttributes() override {
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
         }
      }
   }

   virtual void handleLookup(mlir::Value matched, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matched);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::OpBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::OpBuilder builder3(ifOp.elseRegion());
         builder3.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            auto value = context.getValueForAttribute(refAttr);
            if (refAttr->type != defAttr->type) {
               mlir::Value tmp = builder1.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
               value = tmp;
            }
            context.setValueForAttribute(scope, defAttr, value);
         }
      }
      consumer->consume(this, builder1, context);

      auto trueVal = builder1.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
      builder1.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, trueVal);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
   }

   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::OpBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::OpBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
            context.setValueForAttribute(scope, defAttr, nullValue);
         }
      }
      consumer->consume(this, builder1, context);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      setRequiredBuilderValues(context, ifOp.getResults());
   }

   virtual ~HashOuterJoinLowering() {}
};

class MHashOuterJoinLowering : public mlir::relalg::MarkableHJNode<mlir::relalg::OuterJoinOp> {
   public:
   MHashOuterJoinLowering(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::MarkableHJNode<mlir::relalg::OuterJoinOp>(innerJoinOp, innerJoinOp.left(), innerJoinOp.right()) {
   }
   void addAdditionalRequiredAttributes() override {
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
         }
      }
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
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
      builder1.create<mlir::AtomicRMWOp>(builder1.getUnknownLoc(), builder1.getIntegerType(8), mlir::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            auto value = context.getValueForAttribute(refAttr);
            if (refAttr->type != defAttr->type) {
               mlir::Value tmp = builder1.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
               value = tmp;
            }
            context.setValueForAttribute(scope, defAttr, value);
         }
      }
      consumer->consume(this, builder1, context);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
   }
   virtual void after(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto zero = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(builder.getUnknownLoc(), mlir::arith::CmpIPredicate::eq, marker, zero);
      auto isZeroDB = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), isZero);
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), isZeroDB);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::OpBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::OpBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
            context.setValueForAttribute(scope, defAttr, nullValue);
         }
      }
      consumer->consume(this, builder1, context);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
   }

   virtual ~MHashOuterJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredOuterJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::OuterJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashOuterJoinLowering>(joinOp);
         }
         if (impl.getValue() == "markhash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<MHashOuterJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLOuterJoinLowering>(joinOp);
});