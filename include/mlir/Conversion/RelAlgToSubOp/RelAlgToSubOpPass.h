#ifndef MLIR_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#define MLIR_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createLowerToSubOpPass();
void registerRelAlgToSubOpConversionPasses();
void createLowerRelAlgToSubOpPipeline(mlir::OpPassManager& pm);
} // end namespace relalg
} // end namespace mlir
#endif //MLIR_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
