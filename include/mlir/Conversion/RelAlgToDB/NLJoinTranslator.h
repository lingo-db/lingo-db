#ifndef MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#include "JoinTranslator.h"
#include <mlir/Dialect/DB/IR/DBOps.h>
namespace mlir::relalg {
class NLJoinTranslator : public mlir::relalg::JoinTranslator {
   bool markable;
   size_t vecBuilderId;
   Value vector;
   std::vector<const mlir::relalg::RelationalAttribute*> orderedAttributesLeft;
   mlir::TupleType tupleType;
   mlir::Location loc;

   public:
   NLJoinTranslator(Operator joinOp, Value builderChild, Value lookupChild,bool markable=false) : JoinTranslator(joinOp,builderChild, lookupChild),markable(false),loc(joinOp.getLoc()) {}

   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      addJoinRequiredAttributes();
      addAdditionalRequiredAttributes();
      propagateInfo();
   }

   void build(mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) {
      mlir::Value vectorBuilder = context.builders[vecBuilderId];
      auto const0 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(0));
      mlir::Value packed = markable ? packValues(context, builder, orderedAttributesLeft, {const0}) : packValues(context, builder, orderedAttributesLeft);
      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(loc, vectorBuilder.getType(), vectorBuilder, packed);
      context.builders[vecBuilderId] = mergedBuilder;
   }
   void scanHT(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) {
      auto scope = context.createScope();
      {

         auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypes(context), vector, flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar());
         size_t i = 0;
         for (const auto *attr : orderedAttributesLeft) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         Value marker = markable ? unpacked.getResult(unpacked.getNumResults() - 1) : Value();
         handleScanned(marker, context, builder2);
         builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }

   void handlePotentialMatch(OpBuilder& builder, LoweringContext& context, Value condition) {
      if (condition) {
         auto builderValuesBefore = getRequiredBuilderValues(context);
         auto ifOp = builder.create<mlir::db::IfOp>(
            loc, getRequiredBuilderTypes(context), condition, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            consumer->consume(this, builder1, context);
            builder1.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context)); }, requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(loc, builderValuesBefore); });
         setRequiredBuilderValues(context, ifOp.getResults());
      } else {
         consumer->consume(this, builder, context);
      }
   }

   void probe(mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) {
      auto scope = context.createScope();
      beforeLookup(context, builder);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(loc, getRequiredBuilderTypes(context), vector, flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar());
         size_t i = 0;
         for (const auto *attr : orderedAttributesLeft) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         Value markerLeft = markable ? unpacked.getResult(unpacked.getNumResults() - 1) : Value();
         Value matched=evaluatePredicate(context,builder2,scope);
         handleLookup(matched,markerLeft,context, builder2);
         builder2.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      afterLookup(context,builder);
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      if (child == this->children[0].get()) {
         build(builder, context);
      } else if (child == this->children[1].get()) {
         probe(builder, context);
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      std::vector<mlir::Type> types;
      auto leftAttributes = this->requiredAttributes.intersect(children[0]->getAvailableAttributes());
      for (const auto* attr : leftAttributes) {
         types.push_back(attr->type);
         orderedAttributesLeft.push_back(attr);
      }
      if (markable) {
         types.push_back(mlir::IntegerType::get(op->getContext(), 64));
      }
      tupleType = mlir::TupleType::get(builder.getContext(), types);
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
};
} // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
