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

class HashAntiSemiJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::ProducerConsumerNode* builderChild;
   mlir::relalg::ProducerConsumerNode* lookupChild;


   mlir::relalg::AntiSemiJoinOp joinOp;
   mlir::relalg::Attributes leftKeys, rightKeys, leftValues;
   std::vector<mlir::relalg::RelationalAttribute*> orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;
   size_t builderId;
   mlir::Value joinHt;

   public:
   HashAntiSemiJoinLowering(mlir::relalg::AntiSemiJoinOp antiSemiJoinOp) : mlir::relalg::ProducerConsumerNode({antiSemiJoinOp.left(), antiSemiJoinOp.right()}), joinOp(antiSemiJoinOp) {
      builderChild=children[1].get();
      lookupChild=children[0].get();
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      propagateInfo();
      auto availableLeft = builderChild->getAvailableAttributes();
      auto availableRight = lookupChild->getAvailableAttributes();
      auto [leftKeys, rightKeys, keyTypes] = mlir::relalg::HashJoinUtils::analyzeHJPred(&joinOp.predicate().front(), availableLeft, availableRight);
      this->leftKeys = leftKeys;
      this->rightKeys = rightKeys;
      auto leftValues = availableLeft.intersect(this->requiredAttributes);
      this->leftValues = leftValues;
      std::vector<mlir::Type> valTypes;
      for (auto* x : leftValues) {
         this->orderedValues.push_back(x);
         valTypes.push_back(x->type);
      }
      keyTupleType = mlir::TupleType::get(joinOp.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(joinOp.getContext(), valTypes);
      entryType = mlir::TupleType::get(joinOp.getContext(), {keyTupleType, valTupleType});
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(this->children[0]->getAvailableAttributes());
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == builderChild) {
         auto inlinedKeys = mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), leftKeys, builder.getInsertionBlock(), context);
         mlir::Value packedKey = mlir::relalg::HashJoinUtils::pack(inlinedKeys, builder);
         mlir::Value packedValues =mlir::relalg::HashJoinUtils:: packAttrs(orderedValues, builder, context);
         mlir::Value htBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(joinOp->getLoc(), entryType, mlir::ValueRange({packedKey, packedValues}));
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(joinOp->getLoc(), htBuilder.getType(), htBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      } else if (child == lookupChild) {
         auto packedKey = mlir::relalg::HashJoinUtils::pack(mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), rightKeys, builder.getInsertionBlock(), context), builder);
         mlir::Type htIterable = mlir::db::GenericIterableType::get(builder.getContext(), entryType, "join_ht_iterator");
         mlir::Value matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
         auto matches = builder.create<mlir::db::Lookup>(joinOp->getLoc(), htIterable, joinHt, packedKey);
         {
            auto forOp2 = builder.create<mlir::db::ForOp>(joinOp->getLoc(), mlir::TypeRange{}, matches, matchFoundFlag,mlir::ValueRange());
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(entryType);
            forOp2.getBodyRegion().push_back(block2);
            mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
            auto unpacked = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), entryType.getTypes(), forOp2.getInductionVar()).getResults();
            auto unpackedValue = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), valTupleType.getTypes(), unpacked[1]).getResults();
            for (size_t i = 0; i < unpackedValue.size(); i++) {
               context.setValueForAttribute(scope, orderedValues[i], unpackedValue[i]);
            }
            {
               mlir::relalg::AntiSemiJoinOp clonedInnerJoinOp = mlir::dyn_cast<mlir::relalg::AntiSemiJoinOp>(joinOp->clone());
               mlir::Block* block = &clonedInnerJoinOp.predicate().getBlocks().front();
               auto* terminator = block->getTerminator();

               builder2.mergeRelatinalBlock(block, context, scope);
               mlir::Value matched = mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0];
               builder2.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, matched);
               terminator->erase();
               clonedInnerJoinOp->destroy();
            }
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc());
         }
         {
            mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
            mlir::Value noMatchFound=builder.create<mlir::db::NotOp>(joinOp->getLoc(),mlir::db::BoolType::get(builder.getContext()),matchFound);

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
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto joinHtBuilder = builder.create<mlir::db::CreateJoinHTBuilder>(joinOp.getLoc(), mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      builderId = context.getBuilderId();
      context.builders[builderId] = joinHtBuilder;
      builderChild->addRequiredBuilders({builderId});
      builderChild->produce(context, builder);
      joinHt = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(), mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType), joinHtBuilder);
      lookupChild->produce(context, builder);
   }

   virtual ~HashAntiSemiJoinLowering() {}
};
bool mlir::relalg::ProducerConsumerNodeRegistry::registeredAntiSemiJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::AntiSemiJoinOp joinOp) {
  if (joinOp->hasAttr("impl")) {
     if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
        if (impl.getValue() == "hash") {
           return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashAntiSemiJoinLowering>(joinOp);
        }
     }
  }
  return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLAntiSemiJoinLowering>(joinOp);
});