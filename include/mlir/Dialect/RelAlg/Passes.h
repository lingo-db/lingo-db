#ifndef MLIR_DIALECT_RELALG_PASSES_H
#define MLIR_DIALECT_RELALG_PASSES_H

#include "mlir/Pass/Pass.h"
#include "runtime/Session.h"
#include <memory>
namespace runtime {
class Catalog;
} // end namespace runtime
namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createExtractNestedOperatorsPass();
std::unique_ptr<Pass> createDecomposeLambdasPass();
std::unique_ptr<Pass> createInferNotNullConditionsPass();
std::unique_ptr<Pass> createColumnFoldingPass();
std::unique_ptr<Pass> createImplicitToExplicitJoinsPass();
std::unique_ptr<Pass> createUnnestingPass();
std::unique_ptr<Pass> createPushdownPass();
std::unique_ptr<Pass> createOptimizeJoinOrderPass();
std::unique_ptr<Pass> createCombinePredicatesPass();
std::unique_ptr<Pass> createOptimizeImplementationsPass();
std::unique_ptr<Pass> createIntroduceTmpPass();
std::unique_ptr<Pass> createReduceGroupByKeysPass();
std::unique_ptr<Pass> createExpandTransitiveEqualities();

std::unique_ptr<Pass> createSimplifyAggregationsPass();
std::unique_ptr<Pass> createAttachMetaDataPass(runtime::Catalog& db);
std::unique_ptr<Pass> createDetachMetaDataPass();

std::unique_ptr<Pass> createTrackTuplesPass();

void registerQueryOptimizationPasses();
void setStaticCatalog(std::shared_ptr<runtime::Catalog> catalog);
void createQueryOptPipeline(mlir::OpPassManager& pm, runtime::Catalog* catalog);

} // namespace relalg
} // end namespace mlir

#endif // MLIR_DIALECT_RELALG_PASSES_H