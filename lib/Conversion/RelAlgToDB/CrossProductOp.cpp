#include "mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class CrossProductTranslator : public mlir::relalg::NLJoinTranslator {
   public:
   CrossProductTranslator(mlir::relalg::CrossProductOp crossProductOp) : mlir::relalg::NLJoinTranslator(crossProductOp, crossProductOp.left(), crossProductOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/,mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder,context,matched);
   }
   virtual ~CrossProductTranslator() {}
};
bool mlir::relalg::ProducerConsumerNodeRegistry::registeredCrossProductOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::CrossProductOp crossProductOp) {
   return std::make_unique<CrossProductTranslator>(crossProductOp);
});