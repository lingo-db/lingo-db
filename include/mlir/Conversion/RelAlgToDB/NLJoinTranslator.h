#ifndef MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#include "JoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include <mlir/Dialect/DB/IR/DBOps.h>

namespace mlir::relalg {
class NLJoinTranslator : public mlir::relalg::JoinTranslator {
   Value vector;
   mlir::relalg::OrderedAttributes orderedAttributesLeft;
   mlir::TupleType tupleType;

   protected:
   mlir::Location loc;

   public:
   NLJoinTranslator(std::shared_ptr<JoinImpl> impl) : JoinTranslator(impl), loc(joinOp.getLoc()) {}

   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) override;

   void build(mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context);
   virtual void scanHT(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override;

   void probe(mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context);
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override;
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override;
};
} // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
