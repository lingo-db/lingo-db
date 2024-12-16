#ifndef LINGODB_COMPILER_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
#define LINGODB_COMPILER_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

namespace lingodb::compiler::dialect {
namespace subop {
void setCompressionEnabled(bool compressionEnabled);
std::unique_ptr<mlir::Pass> createLowerSubOpPass();
void registerSubOpToControlFlowConversionPasses();
void createLowerSubOpPipeline(mlir::OpPassManager& pm);
} // end namespace subop
} // end namespace lingodb::compiler::dialect
#endif //LINGODB_COMPILER_CONVERSION_SUBOPTOCONTROLFLOW_SUBOPTOCONTROLFLOWPASS_H
