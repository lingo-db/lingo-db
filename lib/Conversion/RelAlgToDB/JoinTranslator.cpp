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
               mlir::Value tmp = builder.create<mlir::db::AsNullableOp>(joinOp->getLoc(), defAttr->type, value);
               value = tmp;
            }
            context.setValueForAttribute(scope, defAttr, value);
         }
      }
   }
}
void JoinTranslator::handlePotentialMatch(OpBuilder& builder, TranslatorContext& context, Value matches, mlir::function_ref<void(OpBuilder&, TranslatorContext& context, TranslatorContext::AttributeResolverScope&)> onMatch) {
   auto scope = context.createScope();
   builder.create<mlir::scf::IfOp>(
      joinOp->getLoc(), mlir::TypeRange{}, matches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
         if(onMatch){
            onMatch(builder1,context,scope);
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::scf::YieldOp>(joinOp->getLoc(), mlir::ValueRange{}); });
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
