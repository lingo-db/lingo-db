#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Conversion/RelAlgToDB/JoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
namespace {

class LowerToDBPass : public mlir::PassWrapper<LowerToDBPass, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-to-db"; }

   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::util::UtilDialect>();
      registry.insert<mlir::memref::MemRefDialect>();
      registry.insert<mlir::scf::SCFDialect>();
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
   void runOnOperation() override {
      mlir::relalg::TranslatorContext loweringContext;
      getOperation().walk([&](mlir::Operation* op) {
         if (isTranslationHook(op)) {
            auto node = mlir::relalg::Translator::createTranslator(op);
            node->setInfo(nullptr, {});
            mlir::OpBuilder builder(op);
            node->produce(loweringContext, builder);
            node->done();
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createLowerToDBPass() { return std::make_unique<LowerToDBPass>(); }

void registerRelAlgConversionPasses(){
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createLowerToDBPass();
   });

   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-relalg",
      "",
      createLowerRelAlgPipeline);
}
void createLowerRelAlgPipeline(mlir::OpPassManager& pm){
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createLowerToDBPass());
   pm.addPass(mlir::createCanonicalizerPass());
}

} // end namespace relalg
} // end namespace mlir