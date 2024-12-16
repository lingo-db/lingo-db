#ifndef LINGODB_COMPILER_DIALECT_RELALG_PASSES_H
#define LINGODB_COMPILER_DIALECT_RELALG_PASSES_H

#include "lingodb/runtime/Session.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace lingodb::runtime {
class Catalog;
} // end namespace lingodb::runtime

namespace lingodb::compiler::dialect {
namespace relalg {
std::unique_ptr<mlir::Pass> createExtractNestedOperatorsPass();
std::unique_ptr<mlir::Pass> createDecomposeLambdasPass();
std::unique_ptr<mlir::Pass> createInferNotNullConditionsPass();
std::unique_ptr<mlir::Pass> createColumnFoldingPass();
std::unique_ptr<mlir::Pass> createImplicitToExplicitJoinsPass();
std::unique_ptr<mlir::Pass> createUnnestingPass();
std::unique_ptr<mlir::Pass> createPushdownPass();
std::unique_ptr<mlir::Pass> createOptimizeJoinOrderPass();
std::unique_ptr<mlir::Pass> createCombinePredicatesPass();
std::unique_ptr<mlir::Pass> createOptimizeImplementationsPass();
std::unique_ptr<mlir::Pass> createIntroduceTmpPass();
std::unique_ptr<mlir::Pass> createReduceGroupByKeysPass();
std::unique_ptr<mlir::Pass> createExpandTransitiveEqualities();

std::unique_ptr<mlir::Pass> createSimplifyAggregationsPass();
std::unique_ptr<mlir::Pass> createAttachMetaDataPass(runtime::Catalog& db);
std::unique_ptr<mlir::Pass> createDetachMetaDataPass();

std::unique_ptr<mlir::Pass> createTrackTuplesPass();

void registerQueryOptimizationPasses();
void setStaticCatalog(std::shared_ptr<runtime::Catalog> catalog);
void createQueryOptPipeline(mlir::OpPassManager& pm, runtime::Catalog* catalog);

} // namespace relalg
} // end namespace lingodb::compiler::dialect

#endif //LINGODB_COMPILER_DIALECT_RELALG_PASSES_H