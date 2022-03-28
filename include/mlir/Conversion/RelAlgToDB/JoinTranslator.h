#ifndef MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#include "Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include <mlir/Dialect/RelAlg/IR/Column.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h>

namespace mlir::relalg {

class JoinTranslator;
struct JoinImpl {
   virtual mlir::Value getFlag() { return stopOnFlag ? matchFoundFlag : Value(); }
   virtual void addAdditionalRequiredColumns() {}
   virtual void handleLookup(Value matched, Value markerBefore, TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(TranslatorContext& context, mlir::OpBuilder& builder) {}
   virtual void afterLookup(TranslatorContext& context, mlir::OpBuilder& builder) {}
   virtual void handleScanned(Value marker, TranslatorContext& context, mlir::OpBuilder& builder) {}
   virtual void after(TranslatorContext& context, mlir::OpBuilder& builder) {}

   mlir::Value matchFoundFlag;
   bool stopOnFlag = true;
   JoinTranslator* translator;
   mlir::Location loc;
   Operator joinOp;
   Value builderChild, lookupChild;
   bool markable;
   JoinImpl(Operator joinOp, Value builderChild, Value lookupChild, bool markable = false) : loc(joinOp->getLoc()), joinOp(joinOp), builderChild(builderChild), lookupChild(lookupChild), markable(markable) {
   }
};
class JoinTranslator : public Translator {
   protected:
   Operator joinOp;
   mlir::relalg::Translator* builderChild;
   mlir::relalg::Translator* lookupChild;
   std::shared_ptr<JoinImpl> impl;

   public:
   JoinTranslator(std::shared_ptr<JoinImpl> joinImpl);
   void addJoinRequiredColumns();
   void handleMappingNull(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handleMapping(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handlePotentialMatch(OpBuilder& builder, TranslatorContext& context, Value matches, mlir::function_ref<void(OpBuilder&, TranslatorContext& context, TranslatorContext::AttributeResolverScope&)> onMatch = nullptr);

   virtual void scanHT(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   void forwardConsume(mlir::OpBuilder& builder, TranslatorContext& context) {
      consumer->consume(this, builder, context);
   }

   virtual Value evaluatePredicate(TranslatorContext& context, mlir::OpBuilder& builder, TranslatorContext::AttributeResolverScope& scope);
   std::vector<size_t> customLookupBuilders;
};
} // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
