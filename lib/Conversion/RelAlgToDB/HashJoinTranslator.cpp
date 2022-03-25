#include <mlir/Conversion/RelAlgToDB/HashJoinTranslator.h>
using namespace mlir::relalg;
void HashJoinTranslator::setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) {
   this->consumer = consumer;
   this->requiredAttributes = requiredAttributes;
   addJoinRequiredColumns();
   impl->addAdditionalRequiredColumns();
   propagateInfo();
   auto availableLeft = builderChild->getAvailableColumns();
   auto availableRight = lookupChild->getAvailableColumns();
   auto [leftKeys, rightKeys, keyTypes, leftKeyAttributes, canSave] = mlir::relalg::HashJoinUtils::analyzeHJPred(&op->getRegion(0).front(), availableLeft, availableRight);
   this->leftKeys = leftKeys;
   this->rightKeys = rightKeys;
   auto leftValues = availableLeft.intersect(this->requiredAttributes);
   for (size_t i = 0; i < canSave.size(); i++) {
      if (canSave[i]) {
         auto& x = leftKeyAttributes[i];
         leftValues.remove(x);
         orderedKeys.insert(*x.begin());
      } else {
         orderedKeys.insert(nullptr, keyTypes[i]);
      }
   }
   this->orderedValues = mlir::relalg::OrderedAttributes::fromColumns(leftValues);
   keyTupleType = orderedKeys.getTupleType(op.getContext());
   valTupleType = orderedValues.getTupleType(op.getContext(), impl->markable ? std::vector<Type>({mlir::IntegerType::get(op->getContext(), 64)}) : std::vector<Type>());
   entryType = mlir::TupleType::get(op.getContext(), {keyTupleType, valTupleType});
}
void HashJoinTranslator::produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) {
   auto parentPipeline = context.pipelineManager.getCurrentPipeline();
   auto p = std::make_shared<mlir::relalg::Pipeline>(builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>());
   context.pipelineManager.setCurrentPipeline(p);
   context.pipelineManager.addPipeline(p);
   auto res = p->addInitFn([&](mlir::OpBuilder& builder) {
      auto joinHtBuilder = builder.create<mlir::db::HashtableCreate>(loc, mlir::db::JoinHashtableType::get(builder.getContext(), keyTupleType, valTupleType));
      return std::vector<mlir::Value>({joinHtBuilder});
   });
   joinHashtable = p->addDependency(res[0]);
   children[0]->produce(context, p->getBuilder());
   p->finishMainFunction({joinHashtable});
   auto hashtableRes = p->addFinalizeFn([&](mlir::OpBuilder& builder, mlir::ValueRange args) {
      builder.create<mlir::db::HashtableFinalize>(loc, args[0]);
      return std::vector<mlir::Value>{args[0]};
   });
   context.pipelineManager.setCurrentPipeline(parentPipeline);
   joinHt = hashtableRes[0];
   children[1]->produce(context, builder);
   impl->after(context, builder);
   builder.create<mlir::db::FreeOp>(loc, context.pipelineManager.getCurrentPipeline()->addDependency(joinHt));
}

void HashJoinTranslator::unpackValues(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context, Value& marker) {
   auto payloadUnpacked = builder.create<mlir::util::UnPackOp>(loc, packed).getResults();
   if (!valTupleType.getTypes().empty()) {
      auto unpackedValue = impl->markable ? payloadUnpacked.drop_front() : payloadUnpacked;
      orderedValues.setValuesForColumns(context, scope, unpackedValue);
      if (impl->markable) {
         marker = payloadUnpacked[0];
      }
   }
}
void HashJoinTranslator::unpackKeys(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context) {
   mlir::ValueRange unpackedKey = builder.create<mlir::util::UnPackOp>(loc, packed).getResults();
   orderedKeys.setValuesForColumns(context, scope, unpackedKey);
}

void HashJoinTranslator::scanHT(TranslatorContext& context, mlir::OpBuilder& builder) {
   auto scope = context.createScope();
   {
      auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypes(context), context.pipelineManager.getCurrentPipeline()->addDependency(joinHt), context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(entryType, loc);
      block2->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocsCustom(context));
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
      auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar()).getResults();
      unpackKeys(scope, builder2, unpacked[0], context);
      Value marker;
      unpackValues(scope, builder2, unpacked[1], context, marker);
      impl->handleScanned(marker, context, builder2);
      builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context));
      setRequiredBuilderValues(context, forOp2.results());
   }
}
void HashJoinTranslator::consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) {
   auto* ctxt = builder.getContext();
   auto scope = context.createScope();
   if (child == builderChild) {
      auto inlinedKeys = mlir::relalg::HashJoinUtils::inlineKeys(&joinOp->getRegion(0).front(), leftKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context);
      mlir::Value packedKey = builder.create<mlir::util::PackOp>(loc, inlinedKeys);
      auto const0 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(0));
      mlir::Value packedValues = orderedValues.pack(context, builder, loc, impl->markable ? std::vector<Value>{const0} : std::vector<Value>());
      builder.create<mlir::db::HashtableInsert>(loc, joinHashtable, packedKey, packedValues);
   } else if (child == this->children[1].get()) {
      mlir::TupleType entryAndValuePtrType = mlir::TupleType::get(ctxt, TypeRange{entryType, util::RefType::get(ctxt, valTupleType)});
      Type iteratorType = impl->markable ? entryAndValuePtrType : entryType;
      auto packedKey = builder.create<mlir::util::PackOp>(loc, mlir::relalg::HashJoinUtils::inlineKeys(&joinOp->getRegion(0).front(), rightKeys, builder.getInsertionBlock(), builder.getInsertionPoint(), context));
      mlir::Type htIterable = mlir::db::GenericIterableType::get(ctxt, iteratorType, impl->markable ? "join_ht_mod_iterator" : "join_ht_iterator");
      impl->beforeLookup(context, builder);
      auto matches = builder.create<mlir::db::Lookup>(loc, htIterable, context.pipelineManager.getCurrentPipeline()->addDependency(joinHt), packedKey);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypesCustom(context), matches, impl->getFlag(), getRequiredBuilderValuesCustom(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(iteratorType, loc);
         block2->addArguments(getRequiredBuilderTypesCustom(context), getRequiredBuilderLocsCustom(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValuesCustom(context, block2->getArguments().drop_front(1));

         Value entry = forOp2.getInductionVar();
         Value valuePtr;
         if (impl->markable) {
            auto separated = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar()).getResults();
            entry = separated[0];
            valuePtr = separated[1];
         }
         auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, entry).getResults();
         unpackKeys(scope, builder2, unpacked[0], context);
         Value markerVal;
         unpackValues(scope, builder2, unpacked[1], context, markerVal);
         {
            mlir::Value matched = evaluatePredicate(context, builder2, scope);
            Value marker;
            if (impl->markable) {
               Value castedRef = builder2.create<util::GenericMemrefCastOp>(loc, util::RefType::get(ctxt, builder.getI64Type()), valuePtr);
               marker = builder2.create<util::ToMemrefOp>(loc, MemRefType::get({}, builder.getIntegerType(64)), castedRef);
            }
            impl->handleLookup(matched, marker, context, builder2);
         }
         builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValuesCustom(context));
         setRequiredBuilderValuesCustom(context, forOp2.results());
      }
      impl->afterLookup(context, builder);
   }
}