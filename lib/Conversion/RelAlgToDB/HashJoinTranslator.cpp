#include <mlir/Conversion/RelAlgToDB/HashJoinTranslator.h>
using namespace mlir::relalg;
void HashJoinTranslator::setInfo(mlir::relalg::Translator* consumer, mlir::relalg::Attributes requiredAttributes)  {
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
void HashJoinTranslator::produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder)  {
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


void HashJoinTranslator::unpackValues(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context, Value& marker) {
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
void HashJoinTranslator::unpackKeys(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context) {
   mlir::ValueRange unpackedKey = builder.create<mlir::util::UnPackOp>(loc, packed).getResults();
   for (size_t i = 0; i < unpackedKey.size(); i++) {
      if (orderedKeys[i]) {
         context.setValueForAttribute(scope, orderedKeys[i], unpackedKey[i]);
      }
   }
}

void HashJoinTranslator::scanHT(TranslatorContext& context, mlir::OpBuilder& builder) {
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
void HashJoinTranslator::consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context)  {
   auto* ctxt = builder.getContext();
   auto scope = context.createScope();
   if (child == builderChild) {
      auto inlinedKeys = mlir::relalg::HashJoinUtils::inlineKeys(&joinOp->getRegion(0).front(), leftKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context);
      mlir::Value packedKey = builder.create<mlir::util::PackOp>(loc, inlinedKeys);
      auto const0 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(0));
      mlir::Value packedValues = markable ? packValues(context, builder,loc, orderedValues, {const0}) : packValues(context, builder,loc, orderedValues);
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