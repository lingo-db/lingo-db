#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h"

namespace graphalg {

void buildGraphAlgToCorePipeline(mlir::OpPassManager& pm) {
   pm.addPass(createGraphAlgPrepareInline());
   pm.addPass(mlir::createInlinerPass());
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgScalarizeApply());
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgSplitAggregate());
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgToCore());
   pm.addPass(mlir::createCanonicalizerPass());
}
void createGraphAlgToGraphAlgCorePipeline() {
   mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
      "graphalg-to-core-pipeline",
      "Lowers graphalg source IR into core operations",
      buildGraphAlgToCorePipeline);
}
} // namespace graphalg
