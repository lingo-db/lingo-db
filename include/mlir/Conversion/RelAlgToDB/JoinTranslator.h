#ifndef MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#include "ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include <mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h>
#include <mlir/Dialect/RelAlg/IR/RelationalAttribute.h>

namespace mlir::relalg {
class JoinTranslator : public ProducerConsumerNode {
   protected:
   Operator joinOp;
   mlir::relalg::ProducerConsumerNode* builderChild;
   mlir::relalg::ProducerConsumerNode* lookupChild;
   std::vector<size_t> customLookupBuilders;

   JoinTranslator(Operator joinOp, Value builderChild, Value lookupChild) : ProducerConsumerNode({builderChild, lookupChild}), joinOp(joinOp) {
      this->builderChild = children[0].get();
      this->lookupChild = children[1].get();
      this->op = joinOp;
   }
   void handlePotentialMatch(OpBuilder& builder, LoweringContext& context, Value matches) {
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), matches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               consumer->consume(this, builder1, context);
               builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); }, requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
      setRequiredBuilderValues(context, ifOp.getResults());
   }
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
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[1]->addRequiredBuilders(requiredBuilders);
   }

   public:
   virtual void addAdditionalRequiredAttributes() {}
   virtual void handleLookup(Value matched, Value markerBefore, LoweringContext& context, mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(LoweringContext& context, mlir::OpBuilder& builder) {}
   virtual void afterLookup(LoweringContext& context, mlir::OpBuilder& builder) {}
   virtual void handleScanned(Value marker, LoweringContext& context, mlir::OpBuilder& builder) {
   }
   virtual void after(LoweringContext& context, mlir::OpBuilder& builder) {
   }
   virtual Value evaluatePredicate(LoweringContext& context, mlir::OpBuilder& builder, LoweringContext::AttributeResolverScope& scope) {
      bool hasRealPredicate = false;
      if (joinOp->getNumRegions() == 1 && joinOp->getRegion(0).hasOneBlock()) {
         auto terminator = mlir::cast<mlir::relalg::ReturnOp>(joinOp->getRegion(0).front().getTerminator());
         hasRealPredicate = !terminator.results().empty();
      }
      if (hasRealPredicate) {
         return mergeRelationalBlock(
            builder.getInsertionBlock(), joinOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
      } else {
         return builder.create<mlir::db::ConstantOp>(joinOp.getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
      }
   }
   virtual mlir::Value getFlag() { return Value(); }
};
}  // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
