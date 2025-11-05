#ifndef LINGODB_COMPILER_HELPER_H
#define LINGODB_COMPILER_HELPER_H

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace lingodb::compiler {
mlir::LogicalResult applyPatternsGreedily(mlir::Region& region, const mlir::FrozenRewritePatternSet& patterns);
} // namespace lingodb::compiler

#endif //LINGODB_COMPILER_HELPER_H
