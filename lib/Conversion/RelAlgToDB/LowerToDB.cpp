#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilDialect.h"

namespace {

class LowerToDBPass : public mlir::PassWrapper<LowerToDBPass, mlir::FunctionPass> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::util::UtilDialect>();
   }
   bool isTranslationHook(mlir::Operation* op) {
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op)

         .Case<mlir::relalg::MaterializeOp>([&](mlir::Operation* op) {
            return true;
         })
         .Default([&](auto x) {
            return false;
         });
   }
   void runOnFunction() override {
      mlir::relalg::LoweringContext loweringContext;
      loweringContext.executionContext = getFunction().getArgument(0);
      getFunction().walk([&](mlir::Operation* op) {
         if (isTranslationHook(op)) {
            auto node = mlir::relalg::ProducerConsumerNodeRegistry::createNode(op);
            node->setInfo(nullptr, {});
            mlir::relalg::ProducerConsumerBuilder builder(op);
            node->produce(loweringContext, builder);
            node->done();
         }
      });
   }
};
} // end anonymous namespace
mlir::relalg::ProducerConsumerNode::ProducerConsumerNode(mlir::ValueRange potentialChildren) {
   for (auto child : potentialChildren) {
      if (child.getType().isa<mlir::relalg::TupleStreamType>()) {
         children.push_back(mlir::relalg::ProducerConsumerNodeRegistry::createNode(child.getDefiningOp()));
      }
   }
}
namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createLowerToDBPass() { return std::make_unique<LowerToDBPass>(); }
} // end namespace relalg
} // end namespace mlir