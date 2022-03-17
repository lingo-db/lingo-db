#include "mlir/Dialect/SCF/SCF.h"
#include <mlir/Conversion/RelAlgToDB/JoinTranslator.h>
using namespace mlir::relalg;
JoinTranslator::JoinTranslator(std::shared_ptr<JoinImpl> joinImpl) : Translator({joinImpl->builderChild, joinImpl->lookupChild}), joinOp(joinImpl->joinOp), impl(joinImpl) {
   this->builderChild = children[0].get();
   this->lookupChild = children[1].get();
   this->op = joinOp;
   joinImpl->translator = this;
}
void JoinTranslator::addJoinRequiredColumns() {
   this->requiredAttributes.insert(joinOp.getUsedColumns());
   if (joinOp->hasAttr("mapping") && joinOp->getAttr("mapping").isa<mlir::ArrayAttr>()) {
      for (mlir::Attribute attr : joinOp->getAttr("mapping").cast<mlir::ArrayAttr>()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            const auto* refAttr = *mlir::relalg::ColumnSet::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
         }
      }
   }
}
void JoinTranslator::handleMappingNull(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope) {
   if (joinOp->hasAttr("mapping") && joinOp->getAttr("mapping").isa<mlir::ArrayAttr>()) {
      for (mlir::Attribute attr : joinOp->getAttr("mapping").cast<mlir::ArrayAttr>()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         if (this->requiredAttributes.contains(defAttr)) {
            auto nullValue = builder.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
            context.setValueForAttribute(scope, defAttr, nullValue);
         }
      }
   }
}
void JoinTranslator::handleMapping(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope) {
   if (joinOp->hasAttr("mapping") && joinOp->getAttr("mapping").isa<mlir::ArrayAttr>()) {
      for (mlir::Attribute attr : joinOp->getAttr("mapping").cast<mlir::ArrayAttr>()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            const auto* refAttr = *mlir::relalg::ColumnSet::fromArrayAttr(fromExisting).begin();
            auto value = context.getValueForAttribute(refAttr);
            if (refAttr->type != defAttr->type) {
               mlir::Value tmp = builder.create<mlir::db::CastOp>(joinOp->getLoc(), defAttr->type, value);
               value = tmp;
            }
            context.setValueForAttribute(scope, defAttr, value);
         }
      }
   }
}
void JoinTranslator::handlePotentialMatch(OpBuilder& builder, TranslatorContext& context, Value matches, mlir::function_ref<void(OpBuilder&, TranslatorContext& context, TranslatorContext::AttributeResolverScope&)> onMatch) {
   auto scope = context.createScope();
   auto builderValuesBefore = getRequiredBuilderValues(context);
   auto ifOp = builder.create<mlir::scf::IfOp>(
      joinOp->getLoc(), getRequiredBuilderTypes(context), matches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
         if(onMatch){
            onMatch(builder1,context,scope);
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::scf::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); }, requiredBuilders.empty() ? noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::scf::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
   setRequiredBuilderValues(context, ifOp.getResults());
}
std::vector<mlir::Type> JoinTranslator::getRequiredBuilderTypesCustom(TranslatorContext& context) {
   auto requiredBuilderTypes = getRequiredBuilderTypes(context);
   for (auto x : customLookupBuilders) {
      requiredBuilderTypes.push_back(context.builders[x].getType());
   }
   return requiredBuilderTypes;
}
std::vector<mlir::Location> JoinTranslator::getRequiredBuilderLocsCustom(TranslatorContext& context) {
   auto requiredBuilderLocs = getRequiredBuilderLocs(context);
   for (auto x : customLookupBuilders) {
      requiredBuilderLocs.push_back(context.builders[x].getLoc());
   }
   return requiredBuilderLocs;
}
std::vector<mlir::Value> JoinTranslator::getRequiredBuilderValuesCustom(TranslatorContext& context) {
   auto requiredBuilderValues = getRequiredBuilderValues(context);
   for (auto x : customLookupBuilders) {
      requiredBuilderValues.push_back(context.builders[x]);
   }
   return requiredBuilderValues;
}
void JoinTranslator::setRequiredBuilderValuesCustom(TranslatorContext& context, mlir::ValueRange values) {
   size_t i = 0;
   for (auto x : requiredBuilders) {
      context.builders[x] = values[i++];
   }
   for (auto y : customLookupBuilders) {
      context.builders[y] = values[i++];
   }
}
void JoinTranslator::addRequiredBuilders(std::vector<size_t> requiredBuilders) {
   this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   children[1]->addRequiredBuilders(requiredBuilders);
}

mlir::Value JoinTranslator::evaluatePredicate(TranslatorContext& context, mlir::OpBuilder& builder, TranslatorContext::AttributeResolverScope& scope) {
   bool hasRealPredicate = false;
   if (joinOp->getNumRegions() == 1 && joinOp->getRegion(0).hasOneBlock()) {
      auto terminator = mlir::cast<mlir::relalg::ReturnOp>(joinOp->getRegion(0).front().getTerminator());
      hasRealPredicate = !terminator.results().empty();
   }
   if (hasRealPredicate) {
      auto val = mergeRelationalBlock(
         builder.getInsertionBlock(), joinOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
      return builder.create<mlir::db::DeriveTruth>(joinOp->getLoc(), val);
   } else {
      return builder.create<mlir::db::ConstantOp>(joinOp.getLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI64Type(), 1));
   }
}
