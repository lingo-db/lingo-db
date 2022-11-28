#ifndef MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir {
namespace subop {
std::unique_ptr<Pass> createEnforceOrderPass();
std::unique_ptr<Pass> createNormalizeSubOpPass();
std::unique_ptr<Pass> createPullGatherUpPass();
std::unique_ptr<Pass> createReuseLocalPass();
void registerSubOpTransformations();
} // end namespace subop
} // end namespace mlir

#endif //MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
