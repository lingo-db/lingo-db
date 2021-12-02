#ifndef MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#include "Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include <mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h>
#include <mlir/Dialect/RelAlg/IR/RelationalAttribute.h>

namespace mlir::relalg {
class JoinTranslator : public Translator {
   protected:
   Operator joinOp;
   mlir::relalg::Translator* builderChild;
   mlir::relalg::Translator* lookupChild;
   std::vector<size_t> customLookupBuilders;

   JoinTranslator(Operator joinOp, Value builderChild, Value lookupChild);
   void addJoinRequiredAttributes();
   void handleMappingNull(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handleMapping(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handlePotentialMatch(OpBuilder& builder, TranslatorContext& context, Value matches, mlir::function_ref<void(OpBuilder&, TranslatorContext& context, TranslatorContext::AttributeResolverScope&)> onMatch=nullptr);
   std::vector<mlir::Type> getRequiredBuilderTypesCustom(TranslatorContext& context);
   std::vector<mlir::Value> getRequiredBuilderValuesCustom(TranslatorContext& context);
   void setRequiredBuilderValuesCustom(TranslatorContext& context, mlir::ValueRange values);
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override;

   public:
   virtual void addAdditionalRequiredAttributes() {}
   virtual void handleLookup(Value matched, Value markerBefore, TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(TranslatorContext& context, mlir::OpBuilder& builder) {}
   virtual void afterLookup(TranslatorContext& context, mlir::OpBuilder& builder) {}
   virtual void handleScanned(Value marker, TranslatorContext& context, mlir::OpBuilder& builder) {
   }
   virtual void after(TranslatorContext& context, mlir::OpBuilder& builder) {
   }
   virtual Value evaluatePredicate(TranslatorContext& context, mlir::OpBuilder& builder, TranslatorContext::AttributeResolverScope& scope);
   virtual mlir::Value getFlag() { return Value(); }
};
} // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
