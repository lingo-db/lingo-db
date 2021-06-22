#include "mlir/Conversion/RelAlgToDB/HashJoinUtils.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/IR/BlockAndValueMapping.h>

class NLSingleJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::SingleJoinOp joinOp;
   mlir::Value matchFoundFlag;

   public:
   NLSingleJoinLowering(mlir::relalg::SingleJoinOp singleJoinOp) : mlir::relalg::ProducerConsumerNode({singleJoinOp.left(), singleJoinOp.right()}), joinOp(singleJoinOp) {
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
            auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);

            context.setValueForAttribute(scope, defAttr, nullValue);
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, ifOp.getResults());
      } else if (child == this->children[1].get()) {
         mlir::relalg::SingleJoinOp clonedSingleJoinOp = mlir::dyn_cast<mlir::relalg::SingleJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedSingleJoinOp.predicate().getBlocks().front();
         auto* terminator = block->getTerminator();
         mlir::Value condition;
         bool hasCondition = !mlir::cast<mlir::relalg::ReturnOp>(terminator).results().empty();
         if (hasCondition) {
            builder.mergeRelatinalBlock(block, context, scope);
            condition = mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0];
         } else {
            condition = builder.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         }

         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), condition);
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
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            auto value = context.getValueForAttribute(refAttr);
            if (refAttr->type != defAttr->type) {
               mlir::Value tmp = builder1.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
               value = tmp;
            }
            context.setValueForAttribute(scope, defAttr, value);
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
         clonedSingleJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLSingleJoinLowering() {}
};
class ConstantSingleJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::SingleJoinOp joinOp;
   mlir::Value matchFoundFlag;
   std::vector<const mlir::relalg::RelationalAttribute*> attrs;
   std::vector<const mlir::relalg::RelationalAttribute*> origAttrs;
   std::vector<mlir::Type> types;
   size_t builderId;

   public:
   ConstantSingleJoinLowering(mlir::relalg::SingleJoinOp singleJoinOp) : mlir::relalg::ProducerConsumerNode({singleJoinOp.left(), singleJoinOp.right()}), joinOp(singleJoinOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      propagateInfo();
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         attrs.push_back(defAttr);
         types.push_back(defAttr->type);

         auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
         auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
         origAttrs.push_back(refAttr);

      }
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         auto unpacked = builder.create<mlir::util::UnPackOp>(joinOp->getLoc(), types, context.builders[builderId]);
         for (size_t i = 0; i < attrs.size(); i++) {
            context.setValueForAttribute(scope, attrs[i], unpacked.getResult(i));
         }
         consumer->consume(this, builder, context);
      } else if (child == this->children[1].get()) {
         std::vector<mlir::Value> values;
         for (size_t i=0;i<origAttrs.size();i++) {
            mlir::Value value=context.getValueForAttribute(origAttrs[i]);
            if (origAttrs[i]->type != attrs[i]->type) {
               mlir::Value tmp = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), attrs[i]->type, value);
               value = tmp;
            }
            values.push_back(value);
         }
         context.builders[builderId] = builder.create<mlir::util::PackOp>(joinOp->getLoc(), mlir::TupleType::get(builder.getContext(), types), values);
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      std::vector<mlir::Value> values;
      for (auto type : types) {
         values.push_back(builder.create<mlir::db::NullOp>(joinOp.getLoc(), type));
      }
      builderId = context.getBuilderId();
      context.builders[builderId] = builder.create<mlir::util::PackOp>(joinOp->getLoc(), mlir::TupleType::get(builder.getContext(), types), values);
      children[1]->addRequiredBuilders({builderId});
      children[1]->produce(context, builder);
      children[0]->produce(context, builder);
   }

   virtual ~ConstantSingleJoinLowering() {}
};

class HashSingleJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::ProducerConsumerNode* builderChild;
   mlir::relalg::ProducerConsumerNode* lookupChild;
   mlir::relalg::SingleJoinOp joinOp;
   mlir::relalg::Attributes leftKeys, rightKeys, leftValues;
   std::vector<mlir::relalg::RelationalAttribute*> orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;
   size_t builderId;
   mlir::Value joinHt;

   public:
   HashSingleJoinLowering(mlir::relalg::SingleJoinOp singleJoinOp) : mlir::relalg::ProducerConsumerNode({singleJoinOp.left(), singleJoinOp.right()}), joinOp(singleJoinOp) {
      builderChild = children[1].get();
      lookupChild = children[0].get();
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
      return this->builderChild->getAvailableAttributes().insert(this->lookupChild->getAvailableAttributes());
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->builderChild) {
         auto inlinedKeys = mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), leftKeys, builder.getInsertionBlock(), context);
         llvm::dbgs() << "leftKeys:" << (!leftKeys.empty()) << ", inlinedKeys:" << (!inlinedKeys.empty()) << "\n";
         joinOp.dump();
         mlir::Value packedKey = mlir::relalg::HashJoinUtils::pack(inlinedKeys, builder);
         mlir::Value packedValues = mlir::relalg::HashJoinUtils::packAttrs(orderedValues, builder, context);
         mlir::Value htBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(joinOp->getLoc(), entryType, mlir::ValueRange({packedKey, packedValues}));
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(joinOp->getLoc(), htBuilder.getType(), htBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      } else if (child == this->lookupChild) {
         auto packedKey = mlir::relalg::HashJoinUtils::pack(mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), rightKeys, builder.getInsertionBlock(), context), builder);
         mlir::Type htIterable = mlir::db::GenericIterableType::get(builder.getContext(), entryType, "join_ht_iterator");
         mlir::Value matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));

         auto matches = builder.create<mlir::db::Lookup>(joinOp->getLoc(), htIterable, joinHt, packedKey);
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
            for (size_t i = 0; i < unpackedValue.size(); i++) {
               context.setValueForAttribute(scope, orderedValues[i], unpackedValue[i]);
            }
            {
               mlir::relalg::SingleJoinOp clonedSingleJoinOp = mlir::dyn_cast<mlir::relalg::SingleJoinOp>(joinOp->clone());
               mlir::Block* block = &clonedSingleJoinOp.predicate().getBlocks().front();
               auto* terminator = block->getTerminator();
               mlir::Value condition;
               bool hasCondition = !mlir::cast<mlir::relalg::ReturnOp>(terminator).results().empty();
               if (hasCondition) {
                  builder2.mergeRelatinalBlock(block, context, scope);
                  condition = mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0];
               } else {
                  condition = builder2.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
               }

               auto ifOp = builder2.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), condition);
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
                  auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
                  auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
                  auto value = context.getValueForAttribute(refAttr);
                  if (refAttr->type != defAttr->type) {
                     mlir::Value tmp = builder1.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
                     value = tmp;
                  }
                  context.setValueForAttribute(scope, defAttr, value);
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
               clonedSingleJoinOp->destroy();
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
            auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);

            context.setValueForAttribute(scope, defAttr, nullValue);
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, ifOp.getResults());
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto joinHtBuilder = builder.create<mlir::db::CreateJoinHTBuilder>(joinOp.getLoc(), mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      builderId = context.getBuilderId();
      context.builders[builderId] = joinHtBuilder;
      builderChild->addRequiredBuilders({builderId});
      builderChild->produce(context, builder);
      joinHt = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(), mlir::db::TableType::get(builder.getContext()), joinHtBuilder);
      lookupChild->produce(context, builder);
   }

   virtual ~HashSingleJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredSingleJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::SingleJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashSingleJoinLowering>(joinOp);
         } else if (impl.getValue() == "constant") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<ConstantSingleJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLSingleJoinLowering>(joinOp);
});