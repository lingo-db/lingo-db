#ifndef MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#include "JoinTranslator.h"
#include <mlir/Dialect/DB/IR/DBOps.h>
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"

namespace mlir::relalg {
class NLJoinTranslator : public mlir::relalg::JoinTranslator {
   bool markable;
   size_t vecBuilderId;
   Value vector;
   mlir::relalg::OrderedAttributes orderedAttributesLeft;
   mlir::TupleType tupleType;
   protected:
   mlir::Location loc;

   public:
   NLJoinTranslator(Operator joinOp, Value builderChild, Value lookupChild,bool markable=false) : JoinTranslator(joinOp,builderChild, lookupChild),markable(false),loc(joinOp.getLoc()) {}

   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::Attributes requiredAttributes) override;

   void build(mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context);
   void scanHT(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder);


   void probe(mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context);
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override;
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override;
};
} // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
