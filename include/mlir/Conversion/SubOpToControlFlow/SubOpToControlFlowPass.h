#ifndef MLIR_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
#define MLIR_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace subop {
void setCompressionEnabled(bool compressionEnabled);
std::unique_ptr<Pass> createLowerSubOpPass();
void registerSubOpToControlFlowConversionPasses();
void createLowerSubOpPipeline(mlir::OpPassManager& pm);
} // end namespace subop
} // end namespace mlir
#endif //MLIR_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
