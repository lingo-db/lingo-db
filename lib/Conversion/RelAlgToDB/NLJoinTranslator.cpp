#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
using namespace mlir::relalg;
void NLJoinTranslator::setInfo(mlir::relalg::Translator* consumer, mlir::relalg::Attributes requiredAttributes) {
   this->consumer = consumer;
   this->requiredAttributes = requiredAttributes;
   addJoinRequiredAttributes();
   addAdditionalRequiredAttributes();
   propagateInfo();
}

void NLJoinTranslator::build(mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) {
   mlir::Value vectorBuilder = context.builders[vecBuilderId];
   auto const0 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(0));
   mlir::Value packed = orderedAttributesLeft.pack(context, builder, op->getLoc(), markable ? std::vector<Value>{const0} : std::vector<Value>());
   mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(loc, vectorBuilder.getType(), vectorBuilder, packed);
   context.builders[vecBuilderId] = mergedBuilder;
}
void NLJoinTranslator::scanHT(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) {
   auto scope = context.createScope();
   {
      auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypes(context), vector, context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType, loc);
      block2->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocs(context));
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
      auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar());
      orderedAttributesLeft.setValuesForAttributes(context, scope, unpacked.getResults());
      Value marker = markable ? unpacked.getResult(unpacked.getNumResults() - 1) : Value();
      handleScanned(marker, context, builder2);
      builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context));
      setRequiredBuilderValues(context, forOp2.results());
   }
}

void NLJoinTranslator::probe(mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) {
   auto scope = context.createScope();
   beforeLookup(context, builder);
   {
      auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypes(context), vector, getFlag(), getRequiredBuilderValues(context));
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType, loc);
      block2->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocs(context));
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
      auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar());
      orderedAttributesLeft.setValuesForAttributes(context, scope, unpacked.getResults());
      Value markerLeft = markable ? unpacked.getResult(unpacked.getNumResults() - 1) : Value();
      Value matched = evaluatePredicate(context, builder2, scope);
      handleLookup(matched, markerLeft, context, builder2);
      builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context));
      setRequiredBuilderValues(context, forOp2.results());
   }
   afterLookup(context, builder);
}
void NLJoinTranslator::consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) {
   if (child == this->children[0].get()) {
      build(builder, context);
   } else if (child == this->children[1].get()) {
      probe(builder, context);
   }
}
void NLJoinTranslator::produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) {
   auto leftAttributes = this->requiredAttributes.intersect(children[0]->getAvailableAttributes());
   orderedAttributesLeft = mlir::relalg::OrderedAttributes::fromAttributes(leftAttributes);
   tupleType = orderedAttributesLeft.getTupleType(op.getContext(), markable ? std::vector<Type>({mlir::IntegerType::get(op->getContext(), 64)}) : std::vector<Type>());
   mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(loc, mlir::db::VectorBuilderType::get(builder.getContext(), tupleType));
   vecBuilderId = context.getBuilderId();
   context.builders[vecBuilderId] = vectorBuilder;
   children[0]->addRequiredBuilders({vecBuilderId});
   children[0]->produce(context, builder);
   vector = builder.create<mlir::db::BuilderBuild>(loc, mlir::db::VectorType::get(builder.getContext(), tupleType), context.builders[vecBuilderId]);
   children[1]->produce(context, builder);
   after(context, builder);
   builder.create<mlir::db::FreeOp>(loc, vector);
}
