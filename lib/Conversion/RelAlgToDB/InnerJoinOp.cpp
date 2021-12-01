#include "mlir/Conversion/RelAlgToDB/HashJoinUtils.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class NLInnerJoinLowering : public mlir::relalg::NLJoinTranslator {
   public:
   NLInnerJoinLowering(mlir::relalg::InnerJoinOp crossProductOp) : mlir::relalg::NLJoinTranslator(crossProductOp, crossProductOp.left(), crossProductOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/,mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder,context,matched);
   }
   virtual ~NLInnerJoinLowering() {}
};

class HashInnerJoinLowering : public mlir::relalg::HJNode {
   public:
   HashInnerJoinLowering(mlir::relalg::InnerJoinOp innerJoinOp) : mlir::relalg::HJNode(innerJoinOp, innerJoinOp.left(), innerJoinOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/,mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder,context,matched);
   }
   virtual ~HashInnerJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredInnerJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::InnerJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashInnerJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLInnerJoinLowering>(joinOp);
});