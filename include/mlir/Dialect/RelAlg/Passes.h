#ifndef MLIR_DIALECT_RELALG_PASSES_H
#define MLIR_DIALECT_RELALG_PASSES_H

#include "mlir/Pass/Pass.h"
#include "runtime/database.h"
#include <memory>

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createExtractNestedOperatorsPass();
std::unique_ptr<Pass> createDecomposeLambdasPass();
std::unique_ptr<Pass> createImplicitToExplicitJoinsPass();
std::unique_ptr<Pass> createUnnestingPass();
std::unique_ptr<Pass> createPushdownPass();
std::unique_ptr<Pass> createOptimizeJoinOrderPass();
std::unique_ptr<Pass> createCombinePredicatesPass();
std::unique_ptr<Pass> createOptimizeImplementationsPass();
std::unique_ptr<Pass> createIntroduceTmpPass();
std::unique_ptr<Pass> createSimplifyAggregationsPass();
std::unique_ptr<Pass> createAttachMetaDataPass(runtime::Database& db);
std::unique_ptr<Pass> createDetachMetaDataPass();

} // namespace relalg
} // end namespace mlir

#endif // MLIR_DIALECT_RELALG_PASSES_H