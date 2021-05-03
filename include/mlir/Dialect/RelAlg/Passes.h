#ifndef MLIR_DIALECT_RELALG_PASSES_H
#define MLIR_DIALECT_RELALG_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createExtractNestedOperatorsPass();
std::unique_ptr<Pass> createDecomposeLambdasPass();
std::unique_ptr<Pass> createImplicitToExplicitJoinsPass();
std::unique_ptr<Pass> createUnnestingPass();
std::unique_ptr<Pass> createPushdownPass();
std::unique_ptr<Pass> createOptimizeJoinOrderPass();

} // namespace relalg
} // end namespace mlir

#endif // MLIR_DIALECT_RELALG_PASSES_H