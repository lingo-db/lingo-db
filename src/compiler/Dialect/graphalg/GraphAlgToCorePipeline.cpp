#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h"

namespace graphalg {

void buildGraphAlgToCorePipeline(mlir::OpPassManager& pm,
                                 const GraphAlgToCorePipelineOptions& options) {
   pm.addPass(graphalg::createGraphAlgPrepareInline());
   pm.addPass(mlir::createInlinerPass());
   pm.addNestedPass<mlir::func::FuncOp>(
      graphalg::createGraphAlgScalarizeApply());
   pm.addNestedPass<mlir::func::FuncOp>(graphalg::createGraphAlgToCore());
   pm.addPass(mlir::createCanonicalizerPass());
}
void registerGraphAlgToCorePipeline() {
   mlir::PassPipelineRegistration<GraphAlgToCorePipelineOptions>(
      "graphalg-to-core-pipeline",
      "Lowers graphalg source IR into core operations",
      buildGraphAlgToCorePipeline);
}

} // namespace graphalg
