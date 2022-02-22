#include "mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class CrossProductTranslator : public mlir::relalg::JoinImpl {
   public:
   CrossProductTranslator(mlir::relalg::CrossProductOp crossProductOp) : mlir::relalg::JoinImpl(crossProductOp, crossProductOp.left(), crossProductOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched);
   }
   virtual ~CrossProductTranslator() {}
};
std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createCrossProductTranslator(mlir::relalg::CrossProductOp crossProductOp) {
   return std::make_unique<mlir::relalg::NLJoinTranslator>(std::make_shared<CrossProductTranslator>(crossProductOp));
}