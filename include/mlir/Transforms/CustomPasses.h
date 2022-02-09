
#ifndef MLIR_TRANSFORMS_CUSTOMPASSES_H
#define MLIR_TRANSFORMS_CUSTOMPASSES_H
#include "mlir/Pass/Pass.h"

namespace mlir{
std::unique_ptr<Pass> createSinkOpPass();
} // end namespace mlir
#endif // MLIR_TRANSFORMS_CUSTOMPASSES_H
