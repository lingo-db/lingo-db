#ifndef MLIR_CONVERSION_RELALGTODB_HASHJOINUTILS_H
#define MLIR_CONVERSION_RELALGTODB_HASHJOINUTILS_H
#include "ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/Attributes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include <tuple>

namespace mlir::relalg {
class HashJoinUtils {
   public:
   static std::tuple<mlir::relalg::Attributes, mlir::relalg::Attributes, std::vector<mlir::Type>, std::vector<Attributes>> analyzeHJPred(mlir::Block* block, mlir::relalg::Attributes availableLeft, mlir::relalg::Attributes availableRight) {
      llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required;
      mlir::relalg::Attributes leftKeys, rightKeys;
      std::vector<Attributes> leftKeyAttributes;
      std::vector<mlir::Type> types;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::Attributes::from(getAttr.attr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(op)) {
            if (cmpOp.predicate() == mlir::db::DBCmpPredicate::eq && isAndedResult(op)) {
               auto leftAttributes = required[cmpOp.left()];
               auto rightAttributes = required[cmpOp.right()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  leftKeys.insert(leftAttributes);
                  rightKeys.insert(rightAttributes);
                  leftKeyAttributes.push_back(leftAttributes);
                  types.push_back(cmpOp.left().getType());
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  leftKeys.insert(rightAttributes);
                  rightKeys.insert(leftAttributes);
                  leftKeyAttributes.push_back(rightAttributes);
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
      return {leftKeys, rightKeys, types, leftKeyAttributes};
   }

   static bool isAndedResult(mlir::Operation* op, bool first = true) {
      if (mlir::isa<mlir::relalg::ReturnOp>(op)) {
         return true;
      }
      if (mlir::isa<mlir::db::AndOp>(op) || first) {
         for (auto* user : op->getUsers()) {
            if (!isAndedResult(user, false)) return false;
         }
         return true;
      } else {
         return false;
      }
   }
   static std::vector<mlir::Value> inlineKeys(mlir::Block* block, mlir::relalg::Attributes keyAttributes, mlir::Block* newBlock, mlir::Block::iterator insertionPoint, mlir::relalg::LoweringContext& context) {
      llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required;
      mlir::BlockAndValueMapping mapping;
      std::vector<mlir::Value> keys;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::Attributes::from(getAttr.attr())});
            if (keyAttributes.intersects(mlir::relalg::Attributes::from(getAttr.attr()))) {
               mapping.map(getAttr.getResult(), context.getValueForAttribute(&getAttr.attr().getRelationalAttribute()));
            }
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(op)) {
            if (cmpOp.predicate() == mlir::db::DBCmpPredicate::eq && isAndedResult(op)) {
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
                     //todo: remove nasty hack:
                     mlir::OpBuilder builder(cmpOp->getContext());
                     builder.setInsertionPoint(newBlock, insertionPoint);
                     auto helperOp = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexAttr(0));

                     mlir::relalg::detail::inlineOpIntoBlock(keyVal.getDefiningOp(), keyVal.getDefiningOp()->getParentOp(), newBlock->getParentOp(), newBlock, mapping, helperOp);
                     helperOp->remove();
                     helperOp->destroy();
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
};
template <class T>
class HJNode : public mlir::relalg::ProducerConsumerNode {
   protected:
   mlir::relalg::ProducerConsumerNode* builderChild;
   mlir::relalg::ProducerConsumerNode* lookupChild;
   T joinOp;
   mlir::relalg::Attributes leftKeys, rightKeys, leftValues;
   std::vector<const mlir::relalg::RelationalAttribute*> orderedKeys;
   std::vector<const mlir::relalg::RelationalAttribute*> orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;

   std::vector<size_t> customLookupBuilders;
   size_t builderId;
   mlir::Value joinHt;
   HJNode(T joinOp, Value builderChild, Value lookupChild) : ProducerConsumerNode({builderChild, lookupChild}), joinOp(joinOp) {
      this->builderChild = children[0].get();
      this->lookupChild = children[1].get();
      this->op = joinOp;
   }

   public:
   virtual void addAdditionalRequiredAttributes() {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      this->addAdditionalRequiredAttributes();
      propagateInfo();
      auto availableLeft = builderChild->getAvailableAttributes();
      auto availableRight = lookupChild->getAvailableAttributes();
      auto [leftKeys, rightKeys, keyTypes, leftKeyAttributes] = mlir::relalg::HashJoinUtils::analyzeHJPred(&joinOp.predicate().front(), availableLeft, availableRight);
      this->leftKeys = leftKeys;
      this->rightKeys = rightKeys;
      auto leftValues = availableLeft.intersect(this->requiredAttributes);
      for (mlir::relalg::Attributes& x : leftKeyAttributes) {
         if (x.size() == 1) {
            leftValues.remove(x);
            orderedKeys.push_back(*x.begin());
         } else {
            orderedKeys.push_back(nullptr);
         }
      }
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
      return this->joinOp.getAvailableAttributes();
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto joinHtBuilder = builder.create<mlir::db::CreateJoinHTBuilder>(joinOp.getLoc(), mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      builderId = context.getBuilderId();
      context.builders[builderId] = joinHtBuilder;
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      joinHt = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(), mlir::db::JoinHashtableType::get(builder.getContext(), keyTupleType, valTupleType), context.builders[builderId]);
      children[1]->produce(context, builder);
      builder.create<mlir::db::FreeOp>(joinOp->getLoc(), joinHt);
   }
   virtual void handleLookup(Value matched, LoweringContext& context, mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(LoweringContext& context, mlir::OpBuilder& builder) {}
   virtual void afterLookup(LoweringContext& context, mlir::OpBuilder& builder) {}
   virtual mlir::Value getFlag() { return Value(); }

   std::vector<mlir::Type> getRequiredBuilderTypesCustom(LoweringContext& context) {
      auto requiredBuilderTypes = getRequiredBuilderTypes(context);
      for (auto x : customLookupBuilders) {
         requiredBuilderTypes.push_back(context.builders[x].getType());
      }
      return requiredBuilderTypes;
   }
   std::vector<mlir::Value> getRequiredBuilderValuesCustom(LoweringContext& context) {
      auto requiredBuilderValues = getRequiredBuilderValues(context);
      for (auto x : customLookupBuilders) {
         requiredBuilderValues.push_back(context.builders[x]);
      }
      return requiredBuilderValues;
   }
   void setRequiredBuilderValuesCustom(LoweringContext& context, mlir::ValueRange values) {
      size_t i = 0;
      for (auto x : requiredBuilders) {
         context.builders[x] = values[i++];
      }
      for (auto y : customLookupBuilders) {
         context.builders[y] = values[i++];
      }
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == builderChild) {

         auto inlinedKeys = mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), leftKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context);
         mlir::Value packedKey = builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), inlinedKeys);
         mlir::Value packedValues = packValues(context, builder, orderedValues);
         mlir::Value htBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(joinOp->getLoc(), entryType, mlir::ValueRange({packedKey, packedValues}));
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(joinOp->getLoc(), htBuilder.getType(), htBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      } else if (child == this->children[1].get()) {
         auto packedKey =builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), rightKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context));
         mlir::Type htIterable = mlir::db::GenericIterableType::get(builder.getContext(), entryType, "join_ht_iterator");
         beforeLookup(context, builder);
         auto matches = builder.create<mlir::db::Lookup>(joinOp->getLoc(), htIterable, joinHt, packedKey);
         {
            auto forOp2 = builder.create<mlir::db::ForOp>(joinOp->getLoc(), getRequiredBuilderTypesCustom(context), matches, getFlag(), getRequiredBuilderValuesCustom(context));
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(entryType);
            block2->addArguments(getRequiredBuilderTypesCustom(context));
            forOp2.getBodyRegion().push_back(block2);
            mlir::OpBuilder builder2(forOp2.getBodyRegion());
            setRequiredBuilderValuesCustom(context, block2->getArguments().drop_front(1));
            auto unpacked = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), entryType.getTypes(), forOp2.getInductionVar()).getResults();
            mlir::ValueRange unpackedKey = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), keyTupleType.getTypes(), unpacked[0]).getResults();
            for (size_t i = 0; i < unpackedKey.size(); i++) {
               if (orderedKeys[i]) {
                  context.setValueForAttribute(scope, orderedKeys[i], unpackedKey[i]);
               }
            }
            if (!valTupleType.getTypes().empty()) {
               auto unpackedValue = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), valTupleType.getTypes(), unpacked[1]).getResults();
               for (size_t i = 0; i < unpackedValue.size(); i++) {
                  context.setValueForAttribute(scope, orderedValues[i], unpackedValue[i]);
               }
            }
            {
               mlir::Value matched = mergeRelationalBlock(
                  builder2.getInsertionBlock(), joinOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
               handleLookup(matched, context, builder2);
            }
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValuesCustom(context));
            setRequiredBuilderValuesCustom(context, forOp2.results());
         }
         afterLookup(context, builder);
      }
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[1]->addRequiredBuilders(requiredBuilders);
   }
   virtual ~HJNode() {}
};
template <class T>
class MarkableHJNode : public mlir::relalg::ProducerConsumerNode {
   protected:
   mlir::relalg::ProducerConsumerNode* builderChild;
   mlir::relalg::ProducerConsumerNode* lookupChild;
   T joinOp;
   mlir::relalg::Attributes leftKeys, rightKeys, leftValues;
   std::vector<const mlir::relalg::RelationalAttribute*> orderedKeys;
   std::vector<const mlir::relalg::RelationalAttribute*> orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;
   size_t builderId;
   mlir::Value joinHt;
   MarkableHJNode(T joinOp, Value builderChild, Value lookupChild) : ProducerConsumerNode({builderChild, lookupChild}), joinOp(joinOp) {
      this->builderChild = children[0].get();
      this->lookupChild = children[1].get();
      this->op = joinOp;
   }

   public:
   virtual void addAdditionalRequiredAttributes() {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      this->addAdditionalRequiredAttributes();
      propagateInfo();
      auto availableLeft = builderChild->getAvailableAttributes();
      auto availableRight = lookupChild->getAvailableAttributes();
      auto [leftKeys, rightKeys, keyTypes, leftKeyAttributes] = mlir::relalg::HashJoinUtils::analyzeHJPred(&joinOp.predicate().front(), availableLeft, availableRight);
      this->leftKeys = leftKeys;
      this->rightKeys = rightKeys;
      auto leftValues = availableLeft.intersect(this->requiredAttributes);
      for (mlir::relalg::Attributes& x : leftKeyAttributes) {
         if (x.size() == 1) {
            leftValues.remove(x);
            orderedKeys.push_back(*x.begin());
         } else {
            orderedKeys.push_back(nullptr);
         }
      }
      this->leftValues = leftValues;
      std::vector<mlir::Type> valTypes;
      valTypes.push_back(mlir::IntegerType::get(op->getContext(),64));
      for (auto* x : leftValues) {
         this->orderedValues.push_back(x);
         valTypes.push_back(x->type);
      }
      keyTupleType = mlir::TupleType::get(joinOp.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(joinOp.getContext(), valTypes);
      entryType = mlir::TupleType::get(joinOp.getContext(), {keyTupleType, valTupleType});
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->joinOp.getAvailableAttributes();
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto joinHtBuilder = builder.create<mlir::db::CreateJoinHTBuilder>(joinOp.getLoc(), mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      builderId = context.getBuilderId();
      context.builders[builderId] = joinHtBuilder;
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      joinHt = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(), mlir::db::JoinHashtableType::get(builder.getContext(), keyTupleType, valTupleType), context.builders[builderId]);
      children[1]->produce(context, builder);
      after(context, builder);
      builder.create<mlir::db::FreeOp>(joinOp->getLoc(), joinHt);
   }
   virtual void handleLookup(Value matched, Value markerBefore, LoweringContext& context, mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(LoweringContext& context, mlir::OpBuilder& builder) {}
   virtual void afterLookup(LoweringContext& context, mlir::OpBuilder& builder) {}
   virtual void handleScanned(Value marker, LoweringContext& context, mlir::OpBuilder& builder) {
   }
   virtual void after(LoweringContext& context, mlir::OpBuilder& builder) {
   }
   virtual mlir::Value getFlag() { return Value(); }
   void scanHT(LoweringContext& context, mlir::OpBuilder& builder) {
      auto scope = context.createScope();
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), joinHt, this->flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(entryType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(),forOp2.getInductionVar()).getResults();
         mlir::ValueRange unpackedKey = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), keyTupleType.getTypes(), unpacked[0]).getResults();
         for (size_t i = 0; i < unpackedKey.size(); i++) {
            if (orderedKeys[i]) {
               context.setValueForAttribute(scope, orderedKeys[i], unpackedKey[i]);
            }
         }
         auto payloadUnpacked =builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), valTupleType.getTypes(), unpacked[1]).getResults();
         if (!valTupleType.getTypes().empty()) {
            auto unpackedValue = payloadUnpacked.drop_front();
            for (size_t i = 0; i < unpackedValue.size(); i++) {
               context.setValueForAttribute(scope, orderedValues[i], unpackedValue[i]);
            }
         }
         {
            auto marker = payloadUnpacked[0];
            handleScanned(marker, context, builder2);
         }
         builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == builderChild) {
         auto inlinedKeys = mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), leftKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context);
         mlir::Value packedKey = builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), inlinedKeys);
         auto const0 = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerType(64), builder.getI64IntegerAttr(0));

         mlir::Value packedValues = packValues(context, builder, orderedValues,{const0});
         mlir::Value htBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(joinOp->getLoc(), entryType, mlir::ValueRange({packedKey, packedValues}));
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(joinOp->getLoc(), htBuilder.getType(), htBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      } else if (child == this->children[1].get()) {
         mlir::TupleType entryAndValuePtrType = mlir::TupleType::get(builder.getContext(), TypeRange{entryType, util::RefType::get(builder.getContext(), valTupleType,llvm::Optional<int64_t>())});
         auto packedKey = builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), mlir::relalg::HashJoinUtils::inlineKeys(&joinOp.getPredicateBlock(), rightKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context));
         mlir::Type htIterable = mlir::db::GenericIterableType::get(builder.getContext(), entryAndValuePtrType, "join_ht_mod_iterator");
         beforeLookup(context, builder);
         auto matches = builder.create<mlir::db::Lookup>(joinOp->getLoc(), htIterable, joinHt, packedKey);
         {
            auto forOp2 = builder.create<mlir::db::ForOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matches, getFlag(), getRequiredBuilderValues(context));
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(entryAndValuePtrType);
            block2->addArguments(getRequiredBuilderTypes(context));
            forOp2.getBodyRegion().push_back(block2);
            mlir::OpBuilder builder2(forOp2.getBodyRegion());
            setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
            auto separated = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), entryAndValuePtrType.getTypes(), forOp2.getInductionVar()).getResults();

            auto unpacked = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), entryType.getTypes(), separated[0]).getResults();
            mlir::ValueRange unpackedKey = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), keyTupleType.getTypes(), unpacked[0]).getResults();
            for (size_t i = 0; i < unpackedKey.size(); i++) {
               if (orderedKeys[i]) {
                  context.setValueForAttribute(scope, orderedKeys[i], unpackedKey[i]);
               }
            }
            if (!valTupleType.getTypes().empty()) {
               auto unpackedValue = builder2.create<mlir::util::UnPackOp>(joinOp->getLoc(), valTupleType.getTypes(), unpacked[1]).getResults().drop_front();
               for (size_t i = 0; i < unpackedValue.size(); i++) {
                  context.setValueForAttribute(scope, orderedValues[i], unpackedValue[i]);
               }
            }
            {
               mlir::Value matched = mergeRelationalBlock(
                  builder2.getInsertionBlock(), joinOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
               Value castedRef = builder2.create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), util::RefType::get(builder.getContext(), builder.getI64Type(), llvm::Optional<int64_t>()), separated[1]);
               Value asMemRef = builder2.create<util::ToMemrefOp>(builder.getUnknownLoc(), MemRefType::get({}, builder.getIntegerType(64)), castedRef);

               handleLookup(matched, asMemRef, context, builder2);
            }
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
            setRequiredBuilderValues(context, forOp2.results());
         }
         afterLookup(context, builder);
      }
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[1]->addRequiredBuilders(requiredBuilders);
   }
   virtual ~MarkableHJNode() {}
};
} // end namespace mlir::relalg

#endif // MLIR_CONVERSION_RELALGTODB_HASHJOINUTILS_H
