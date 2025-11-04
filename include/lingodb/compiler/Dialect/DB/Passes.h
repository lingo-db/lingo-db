
#ifndef LINGODB_COMPILER_DIALECT_DB_PASSES_H
#define LINGODB_COMPILER_DIALECT_DB_PASSES_H
#include "mlir/Pass/Pass.h"

namespace lingodb::compiler::dialect::db {
std::unique_ptr<mlir::Pass> createPrepareLoweringPass();
std::unique_ptr<mlir::Pass> createSimplifyToArithPass();
void addOptimizeRuntimeFunctionPatterns(mlir::RewritePatternSet& patterns);
} // end namespace lingodb::compiler::dialect::db
#endif //LINGODB_COMPILER_DIALECT_DB_PASSES_H
