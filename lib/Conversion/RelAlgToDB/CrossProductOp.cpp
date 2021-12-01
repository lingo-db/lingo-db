#include "mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class CrossProductLowering : public mlir::relalg::NLJoinTranslator<mlir::relalg::CrossProductOp> {
   public:
   CrossProductLowering(mlir::relalg::CrossProductOp crossProductOp) : mlir::relalg::NLJoinTranslator<mlir::relalg::CrossProductOp>(crossProductOp, crossProductOp.left(), crossProductOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/,mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder,context,matched);
   }
   virtual ~CrossProductLowering() {}
};
bool mlir::relalg::ProducerConsumerNodeRegistry::registeredCrossProductOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::CrossProductOp crossProductOp) {
   return std::make_unique<CrossProductLowering>(crossProductOp);
});