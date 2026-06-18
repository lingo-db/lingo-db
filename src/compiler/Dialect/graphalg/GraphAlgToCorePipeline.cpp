#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h"

namespace graphalg {

void buildGraphAlgToCorePipeline(mlir::OpPassManager& pm) {
   pm.addPass(createGraphAlgPrepareInline());
   pm.addPass(mlir::createInlinerPass());
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgScalarizeApply());
   // Fuse co-operand matmuls (A*X + Bᵀ*X -> (A+B)*X) before they are decomposed
   // into mxm_join/deferred_reduce, so undirected propagation runs one matmul.
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgFuseMatMul());
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgSplitAggregate());
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgToCore());
   pm.addPass(mlir::createCanonicalizerPass());
   // Rewrite eligible fixpoint loops (WCC/SSSP-style) into semi-naive form so
   // each iteration propagates only the changed delta instead of the full state.
   pm.addNestedPass<mlir::func::FuncOp>(createGraphAlgSemiNaive());
   pm.addPass(mlir::createCanonicalizerPass());
}
void createGraphAlgToGraphAlgCorePipeline() {
   mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
      "graphalg-to-core-pipeline",
      "Lowers graphalg source IR into core operations",
      buildGraphAlgToCorePipeline);
}
} // namespace graphalg
