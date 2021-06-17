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
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         children[1]->produce(context, builder);
      } else if (child == this->children[1].get()) {
         mlir::relalg::InnerJoinOp clonedInnerJoinOp = mlir::dyn_cast<mlir::relalg::InnerJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedInnerJoinOp.predicate().getBlocks().front();
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
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

         size_t i = 0;
         for (auto b : requiredBuilders) {
            context.builders[b] = ifOp.getResult(i++);
         }
         terminator->erase();
         clonedInnerJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLInnerJoinLowering() {}
};

std::tuple<mlir::relalg::Attributes, mlir::relalg::Attributes,std::vector<mlir::Type>> analyzeHJPred(mlir::Block* block, mlir::relalg::Attributes availableLeft, mlir::relalg::Attributes availableRight) {
   llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required;
   mlir::relalg::Attributes leftKeys, rightKeys;
   std::vector<mlir::Type> types;
   block->walk([&](mlir::Operation* op) {
      if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
         required.insert({getAttr.getResult(), mlir::relalg::Attributes::from(getAttr.attr())});
      } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(op)) {
         if (cmpOp.predicate() == mlir::db::DBCmpPredicate::eq) {
            auto leftAttributes = required[cmpOp.left()];
            auto rightAttributes = required[cmpOp.right()];
            if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
               leftKeys.insert(leftAttributes);
               rightKeys.insert(rightAttributes);
               types.push_back(cmpOp.left().getType());
            } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
               leftKeys.insert(rightAttributes);
               rightKeys.insert(leftAttributes);
               types.push_back(cmpOp.left().getType());
            }
         }
      } else {
         mlir::relalg::Attributes attributes;
         for (auto operand : op->getOperands()) {
            if (required.count(operand)) {
               attributes.insert(required[operand]);
            }
         }
         for (auto result : op->getResults()) {
            required.insert({result, attributes});
         }
      }
   });
   return {leftKeys, rightKeys,types};
}
mlir::Value pack(std::vector<mlir::Value> values, mlir::OpBuilder& builder) {
   std::vector<mlir::Type> types;
   for (auto v : values) {
      types.push_back(v.getType());
   }
   auto tupleType = mlir::TupleType::get(builder.getContext(), types);
   return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), tupleType, values);
}
mlir::Value packAttrs(std::vector<mlir::relalg::RelationalAttribute*> attrs, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) {
   std::vector<mlir::Type> types;
   std::vector<mlir::Value> values;
   for (auto *attr : attrs) {
      auto v = context.getValueForAttribute(attr);
      types.push_back(v.getType());
      values.push_back(v);
   }
   auto tupleType = mlir::TupleType::get(builder.getContext(), types);
   return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), tupleType, values);
}
std::vector<mlir::Value> inlineKeys(mlir::Block* block, mlir::relalg::Attributes keyAttributes, mlir::Block* newBlock, mlir::relalg::LoweringContext& context) {
   llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required;
   mlir::BlockAndValueMapping mapping;
   std::vector<mlir::Value> keys;
   block->walk([&](mlir::Operation* op) {
      if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
         required.insert({getAttr.getResult(), mlir::relalg::Attributes::from(getAttr.attr())});
         if (keyAttributes.intersects(mlir::relalg::Attributes::from(getAttr.attr()))) {
            llvm::dbgs() << "\ncould not find:";
            mlir::relalg::Attributes::from(getAttr.attr()).dump(getAttr->getContext());
            llvm::dbgs() << "\n";
            mapping.map(getAttr.getResult(), context.getValueForAttribute(&getAttr.attr().getRelationalAttribute()));
         }
      } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(op)) {
         if (cmpOp.predicate() == mlir::db::DBCmpPredicate::eq) {
            auto leftAttributes = required[cmpOp.left()];
            auto rightAttributes = required[cmpOp.right()];
            mlir::Value keyVal;
            if (leftAttributes.isSubsetOf(keyAttributes)) {
               keyVal = cmpOp.left();
            } else if (rightAttributes.isSubsetOf(keyAttributes)) {
               keyVal = cmpOp.right();
            }
            if (keyVal) {
               if (!mapping.contains(keyVal)) {
                  mlir::relalg::detail::inlineOpIntoBlock(keyVal.getDefiningOp(), keyVal.getDefiningOp()->getParentOp(), newBlock->getParentOp(), newBlock, mapping);
               }
               keys.push_back(mapping.lookupOrNull(keyVal));
            }
         }
      } else {
         mlir::relalg::Attributes attributes;
         for (auto operand : op->getOperands()) {
            if (required.count(operand)) {
               attributes.insert(required[operand]);
            }
         }
         for (auto result : op->getResults()) {
            required.insert({result, attributes});
         }
      }
   });
   return keys;
}
class HashInnerJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::InnerJoinOp joinOp;
   mlir::relalg::Attributes leftKeys, rightKeys, leftValues;
   std::vector<mlir::relalg::RelationalAttribute*> orderedValues;
   mlir::TupleType keyTupleType,valTupleType, entryType;
   size_t builderId;
   mlir::Value joinHt;
   public:
   HashInnerJoinLowering(mlir::relalg::InnerJoinOp innerJoinOp) : mlir::relalg::ProducerConsumerNode({innerJoinOp.left(), innerJoinOp.right()}), joinOp(innerJoinOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      propagateInfo();
      auto availableLeft = children[0]->getAvailableAttributes();
      auto availableRight = children[1]->getAvailableAttributes();
      auto [leftKeys, rightKeys,keyTypes] = analyzeHJPred(&joinOp.predicate().front(), availableLeft, availableRight);
      this->leftKeys = leftKeys;
      this->rightKeys = rightKeys;
      llvm::dbgs() << "leftKeys:";
      leftKeys.dump(joinOp.getContext());
      llvm::dbgs() << "\n rightKeys:";
      rightKeys.dump(joinOp.getContext());
      auto leftValues = availableLeft.intersect(this->requiredAttributes);
      this->leftValues = leftValues;
      llvm::dbgs() << "leftValues:";
      leftValues.dump(joinOp.getContext());
      llvm::dbgs() << "\n requiredAttributes:";
      this->requiredAttributes.dump(joinOp.getContext());
      llvm::dbgs() << "\n";
      std::vector<mlir::Type> valTypes;
      for (auto *x : leftValues) {
         this->orderedValues.push_back(x);
         valTypes.push_back(x->type);
      }
      keyTupleType=mlir::TupleType::get(joinOp.getContext(),keyTypes);
      valTupleType=mlir::TupleType::get(joinOp.getContext(),valTypes);
      entryType=mlir::TupleType::get(joinOp.getContext(), {keyTupleType,valTupleType});

   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(this->children[1]->getAvailableAttributes());
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         mlir::Value packedKey = pack(inlineKeys(&joinOp.getPredicateBlock(), leftKeys, builder.getInsertionBlock(), context), builder);
         mlir::Value packedValues = packAttrs(orderedValues, builder, context);
         mlir::Value htBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(joinOp->getLoc(), entryType, mlir::ValueRange({packedKey,packedValues}));
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(joinOp->getLoc(), htBuilder.getType(), htBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      } else if (child == this->children[1].get()) {
         auto packedKey = pack(inlineKeys(&joinOp.getPredicateBlock(), rightKeys, builder.getInsertionBlock(), context), builder);
         mlir::Type htIterable = mlir::db::GenericIterableType::get(builder.getContext(), entryType, "join_ht_iterator");
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
               mlir::relalg::InnerJoinOp clonedInnerJoinOp = mlir::dyn_cast<mlir::relalg::InnerJoinOp>(joinOp->clone());
               mlir::Block* block = &clonedInnerJoinOp.predicate().getBlocks().front();
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
               consumer->consume(this, builder1, context);
               builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

               size_t i = 0;
               for (auto b : requiredBuilders) {
                  context.builders[b] = ifOp.getResult(i++);
               }
               terminator->erase();
               clonedInnerJoinOp->destroy();
            }
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
            setRequiredBuilderValues(context, forOp2.results());
         }
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto joinHtBuilder = builder.create<mlir::db::CreateJoinHTBuilder>(joinOp.getLoc(), mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      builderId=context.getBuilderId();
      context.builders[builderId]=joinHtBuilder;
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      joinHt = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(), mlir::db::TableType::get(builder.getContext()), joinHtBuilder);
      children[1]->produce(context, builder);
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