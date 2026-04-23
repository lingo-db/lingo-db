#include "lingodb/compiler/helper.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h"

namespace graphalg {
using namespace mlir;

void createLowerGraphAlgCoreToRelAlgPipeline(mlir::OpPassManager& pm) {
   pm.addPass(createGraphAlgToRelAlgPass());
   pm.addPass(lingodb::compiler::createCanonicalizerPass());
}
void registerGraphAlgCoreToRelAlgConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return createGraphAlgToRelAlgPass();
   });
   mlir::PassPipelineRegistration<>(
      "lower-graphalg-core-to-relalg",
      "",
      createLowerGraphAlgCoreToRelAlgPipeline);
}
}