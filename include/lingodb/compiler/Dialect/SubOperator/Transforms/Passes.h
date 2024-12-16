#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace lingodb::compiler::dialect {
namespace subop {
std::unique_ptr<mlir::Pass> createFoldColumnsPass();
std::unique_ptr<mlir::Pass> createEnforceOrderPass();
std::unique_ptr<mlir::Pass> createNormalizeSubOpPass();
std::unique_ptr<mlir::Pass> createSpecializeSubOpPass(bool withOptimizations);
std::unique_ptr<mlir::Pass> createPullGatherUpPass();
std::unique_ptr<mlir::Pass> createReuseLocalPass();
std::unique_ptr<mlir::Pass> createGlobalOptPass();
std::unique_ptr<mlir::Pass> createParallelizePass();
std::unique_ptr<mlir::Pass> createSpecializeParallelPass();
std::unique_ptr<mlir::Pass> createSplitIntoExecutionStepsPass();
std::unique_ptr<mlir::Pass> createInlineNestedMapPass();
std::unique_ptr<mlir::Pass> createFinalizePass();
std::unique_ptr<mlir::Pass> createPrepareLoweringPass();
void registerSubOpTransformations();
} // end namespace subop
} // end namespace lingodb::compiler::dialect

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_PASSES_H
