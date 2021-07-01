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
         if(this->requiredAttributes.contains(defAttr)) {
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
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
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
         for (mlir::Attribute attr : joinOp.mapping()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if(this->requiredAttributes.contains(defAttr)) {
               auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);

               context.setValueForAttribute(scope, defAttr, nullValue);
            }
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context,ifOp.getResults());
      } else if (child == this->children[1].get()) {
         mlir::relalg::OuterJoinOp clonedOuterJoinOp = mlir::dyn_cast<mlir::relalg::OuterJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedOuterJoinOp.predicate().getBlocks().front();
         auto* terminator = block->getTerminator();

         builder.mergeRelatinalBlock(block, context, scope);

         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0]);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::relalg::ProducerConsumerBuilder builder2(ifOp.elseRegion());
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         for (mlir::Attribute attr : joinOp.mapping()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if(this->requiredAttributes.contains(defAttr)) {
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
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLOuterJoinLowering() {}
};

class HashOuterJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::ProducerConsumerNode* builderChild;
   mlir::relalg::ProducerConsumerNode* lookupChild;
   mlir::relalg::OuterJoinOp joinOp;
   mlir::relalg::Attributes leftKeys, rightKeys, leftValues;
   std::vector<mlir::relalg::RelationalAttribute*> orderedValues;
   mlir::TupleType keyTupleType,valTupleType, entryType;
   size_t builderId;
   mlir::Value joinHt;
   public:
   HashOuterJoinLowering(mlir::relalg::OuterJoinOp outerJoinOp) : mlir::relalg::ProducerConsumerNode({outerJoinOp.left(), outerJoinOp.right()}), joinOp(outerJoinOp) {
      builderChild=children[1].get();
      lookupChild=children[0].get();
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if(this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
         }
      }
      propagateInfo();
      auto availableLeft = builderChild->getAvailableAttributes();
      auto availableRight = lookupChild->getAvailableAttributes();
      auto [leftKeys, rightKeys,keyTypes] = mlir::relalg::HashJoinUtils::analyzeHJPred(&joinOp.predicate().front(), availableLeft, availableRight);
      this->leftKeys = leftKeys;
      this->rightKeys = rightKeys;
      auto leftValues = availableLeft.intersect(this->requiredAttributes);
      this->leftValues = leftValues;
      std::vector<mlir::Type> valTypes;
      for (auto *x : leftValues) {
         this->orderedValues.push_back(x);
         valTypes.push_back(x->type);
      }
      keyTupleType=mlir::TupleType::get(joinOp.getContext(),keyTypes);
      valTupleType=mlir::TupleType::get(joinOp.getContext(),valTypes);
      entryType=mlir::TupleType::get(joinOp.getContext(), {keyTupleType,valTupleType});

   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      lookupChild->addRequiredBuilders(requiredBuilders);
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return joinOp.getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->builderChild) {
         auto inlinedKeys=mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), leftKeys, builder.getInsertionBlock(), context);
         mlir::Value packedKey = mlir::relalg::HashJoinUtils::pack(inlinedKeys, builder);
         mlir::Value packedValues = mlir::relalg::HashJoinUtils::packAttrs(orderedValues, builder, context);
         mlir::Value htBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(joinOp->getLoc(), entryType, mlir::ValueRange({packedKey,packedValues}));
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(joinOp->getLoc(), htBuilder.getType(), htBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      } else if (child == this->lookupChild) {
         auto packedKey = mlir::relalg::HashJoinUtils::pack(mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), rightKeys, builder.getInsertionBlock(), context), builder);
         mlir::Type htIterable = mlir::db::GenericIterableType::get(builder.getContext(), entryType, "join_ht_iterator");
         mlir::Value matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));

         auto matches=builder.create<mlir::db::Lookup>(joinOp->getLoc(),htIterable,joinHt,packedKey);
         {
            auto forOp2 = builder.create<mlir::db::ForOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matches, mlir::Value(), getRequiredBuilderValues(context));
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(entryType);
            block2->addArguments(getRequiredBuilderTypes(context));
            forOp2.getBodyRegion().push_back(block2);
            mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
            setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
            auto unpacked = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), entryType.getTypes(), forOp2.getInductionVar()).getResults();
            //mlir::ValueRange unpackedKey= builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), keyTupleType.getTypes(), unpacked[0]).getResults();
            auto unpackedValue = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), valTupleType.getTypes(), unpacked[1]).getResults();
            for(size_t i=0;i<unpackedValue.size();i++) {
               context.setValueForAttribute(scope,orderedValues[i],unpackedValue[i]);
            }
            {
               mlir::relalg::OuterJoinOp clonedOuterJoinOp = mlir::dyn_cast<mlir::relalg::OuterJoinOp>(joinOp->clone());
               mlir::Block* block = &clonedOuterJoinOp.predicate().getBlocks().front();
               auto* terminator = block->getTerminator();

               builder2.mergeRelatinalBlock(block, context, scope);

               auto ifOp = builder2.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0]);
               mlir::Block* ifBlock = new mlir::Block;

               ifOp.thenRegion().push_back(ifBlock);

               mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
               if (!requiredBuilders.empty()) {
                  mlir::Block* elseBlock = new mlir::Block;
                  ifOp.elseRegion().push_back(elseBlock);
                  mlir::relalg::ProducerConsumerBuilder builder3(ifOp.elseRegion());
                  builder3.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
               }
               for (mlir::Attribute attr : joinOp.mapping()) {
                  auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
                  auto* defAttr = &relationDefAttr.getRelationalAttribute();
                  if(this->requiredAttributes.contains(defAttr)) {
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
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
            setRequiredBuilderValues(context, forOp2.results());
         }
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
         for (mlir::Attribute attr : joinOp.mapping()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if(this->requiredAttributes.contains(defAttr)) {
               auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
               context.setValueForAttribute(scope, defAttr, nullValue);
            }
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context,ifOp.getResults());
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto joinHtBuilder = builder.create<mlir::db::CreateJoinHTBuilder>(joinOp.getLoc(), mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      builderId=context.getBuilderId();
      context.builders[builderId]=joinHtBuilder;
      builderChild->addRequiredBuilders({builderId});
      builderChild->produce(context, builder);
      joinHt = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(),mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType), context.builders[builderId]);
      lookupChild->produce(context, builder);
   }

   virtual ~HashOuterJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredOuterJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::OuterJoinOp joinOp) {
  if (joinOp->hasAttr("impl")) {
     if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
        if (impl.getValue() == "hash") {
           return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashOuterJoinLowering>(joinOp);
        }
     }
  }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLOuterJoinLowering>(joinOp);
});