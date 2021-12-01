#ifndef MLIR_CONVERSION_RELALGTODB_HASHJOINUTILS_H
#define MLIR_CONVERSION_RELALGTODB_HASHJOINUTILS_H
#include "JoinTranslator.h"
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

class HJNode : public mlir::relalg::JoinTranslator {
   protected:
   mlir::Location loc;
   bool markable;
   mlir::relalg::Attributes leftKeys, rightKeys;
   std::vector<const mlir::relalg::RelationalAttribute*> orderedKeys;
   std::vector<const mlir::relalg::RelationalAttribute*> orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;
   size_t builderId;
   mlir::Value joinHt;

   HJNode(Operator joinOp, Value builderChild, Value lookupChild,bool markable = false) : JoinTranslator(joinOp,builderChild, lookupChild), loc(joinOp.getLoc()),markable(markable) {}

   public:

   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      addJoinRequiredAttributes();
      this->addAdditionalRequiredAttributes();
      propagateInfo();
      auto availableLeft = builderChild->getAvailableAttributes();
      auto availableRight = lookupChild->getAvailableAttributes();
      auto [leftKeys, rightKeys, keyTypes, leftKeyAttributes] = mlir::relalg::HashJoinUtils::analyzeHJPred(&op->getRegion(0).front(), availableLeft, availableRight);
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
      std::vector<mlir::Type> valTypes;
      if (markable) {
         valTypes.push_back(mlir::IntegerType::get(op->getContext(), 64));
      }
      for (const auto* x : leftValues) {
         this->orderedValues.push_back(x);
         valTypes.push_back(x->type);
      }
      keyTupleType = mlir::TupleType::get(op.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(op.getContext(), valTypes);
      entryType = mlir::TupleType::get(op.getContext(), {keyTupleType, valTupleType});
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto joinHtBuilder = builder.create<mlir::db::CreateJoinHTBuilder>(loc, mlir::db::JoinHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      builderId = context.getBuilderId();
      context.builders[builderId] = joinHtBuilder;
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      joinHt = builder.create<mlir::db::BuilderBuild>(loc, mlir::db::JoinHashtableType::get(builder.getContext(), keyTupleType, valTupleType), context.builders[builderId]);
      children[1]->produce(context, builder);
      after(context, builder);
      builder.create<mlir::db::FreeOp>(loc, joinHt);
   }


   void unpackValues(LoweringContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, LoweringContext& context, Value& marker) {
      auto payloadUnpacked = builder.create<mlir::util::UnPackOp>(loc, packed).getResults();
      if (!valTupleType.getTypes().empty()) {
         auto unpackedValue = markable ? payloadUnpacked.drop_front() : payloadUnpacked;
         for (size_t i = 0; i < unpackedValue.size(); i++) {
            context.setValueForAttribute(scope, orderedValues[i], unpackedValue[i]);
         }
         if (markable) {
            marker = payloadUnpacked[0];
         }
      }
   }
   void unpackKeys(LoweringContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, LoweringContext& context) {
      mlir::ValueRange unpackedKey = builder.create<mlir::util::UnPackOp>(loc, packed).getResults();
      for (size_t i = 0; i < unpackedKey.size(); i++) {
         if (orderedKeys[i]) {
            context.setValueForAttribute(scope, orderedKeys[i], unpackedKey[i]);
         }
      }
   }

   void scanHT(LoweringContext& context, mlir::OpBuilder& builder) {
      auto scope = context.createScope();
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypes(context), joinHt, this->flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(entryType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar()).getResults();
         unpackKeys(scope, builder2, unpacked[0], context);
         Value marker;
         unpackValues(scope, builder2, unpacked[1], context, marker);
         handleScanned(marker, context, builder2);
         builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto* ctxt = builder.getContext();
      auto scope = context.createScope();
      if (child == builderChild) {
         auto inlinedKeys = mlir::relalg::HashJoinUtils::inlineKeys(&joinOp->getRegion(0).front(), leftKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context);
         mlir::Value packedKey = builder.create<mlir::util::PackOp>(loc, inlinedKeys);
         auto const0 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(0));
         mlir::Value packedValues = markable ? packValues(context, builder, orderedValues, {const0}) : packValues(context, builder, orderedValues);
         mlir::Value htBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(loc, mlir::ValueRange({packedKey, packedValues}));
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(loc, htBuilder.getType(), htBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      } else if (child == this->children[1].get()) {
         mlir::TupleType entryAndValuePtrType = mlir::TupleType::get(ctxt, TypeRange{entryType, util::RefType::get(ctxt, valTupleType, llvm::Optional<int64_t>())});
         Type iteratorType = markable ? entryAndValuePtrType : entryType;
         auto packedKey = builder.create<mlir::util::PackOp>(loc, mlir::relalg::HashJoinUtils::inlineKeys(&joinOp->getRegion(0).front(), rightKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context));
         mlir::Type htIterable = mlir::db::GenericIterableType::get(ctxt, iteratorType, markable ? "join_ht_mod_iterator" : "join_ht_iterator");
         beforeLookup(context, builder);
         auto matches = builder.create<mlir::db::Lookup>(loc, htIterable, joinHt, packedKey);
         {
            auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypesCustom(context), matches, getFlag(), getRequiredBuilderValuesCustom(context));
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(iteratorType);
            block2->addArguments(getRequiredBuilderTypesCustom(context));
            forOp2.getBodyRegion().push_back(block2);
            mlir::OpBuilder builder2(forOp2.getBodyRegion());
            setRequiredBuilderValuesCustom(context, block2->getArguments().drop_front(1));

            Value entry = forOp2.getInductionVar();
            Value valuePtr;
            if (markable) {
               auto separated = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar()).getResults();
               entry = separated[0];
               valuePtr = separated[1];
            }
            auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, entry).getResults();
            unpackKeys(scope, builder2, unpacked[0], context);
            Value markerVal;
            unpackValues(scope, builder2, unpacked[1], context, markerVal);
            {
               mlir::Value matched = evaluatePredicate(context,builder2,scope);
               Value marker;
               if (markable) {
                  Value castedRef = builder2.create<util::GenericMemrefCastOp>(loc, util::RefType::get(ctxt, builder.getI64Type(), llvm::Optional<int64_t>()), valuePtr);
                  marker = builder2.create<util::ToMemrefOp>(loc, MemRefType::get({}, builder.getIntegerType(64)), castedRef);
               }
               handleLookup(matched, marker, context, builder2);
            }
            builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValuesCustom(context));
            setRequiredBuilderValuesCustom(context, forOp2.results());
         }
         afterLookup(context, builder);
      }
   }

   virtual ~HJNode() {}
};
} // end namespace mlir::relalg

#endif // MLIR_CONVERSION_RELALGTODB_HASHJOINUTILS_H
