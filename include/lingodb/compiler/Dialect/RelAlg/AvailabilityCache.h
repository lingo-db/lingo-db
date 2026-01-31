#ifndef LINGODB_COMPILER_DIALECT_RELALG_AVAILABILITYCACHE_H
#define LINGODB_COMPILER_DIALECT_RELALG_AVAILABILITYCACHE_H

#include "lingodb/compiler/Dialect/RelAlg/ColumnSet.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"

namespace lingodb::compiler::dialect::relalg {

// Operator is already defined when this header is included (via RelAlgOpsInterfaces.h)

/// Cache for memoizing getAvailableColumns() results.
/// This avoids redundant computation when traversing operator trees,
/// as available column computation typically recurses into child operators.
class AvailabilityCache {
   llvm::DenseMap<mlir::Operation*, ColumnSet> cache;

   public:
   /// Get available columns for an operator, using cached result if available.
   /// If not cached, computes and caches the result.
   /// Implementation in AvailabilityCache.cpp
   ColumnSet getAvailableColumnsFor(Operator op);

   /// Clear the cache.
   void clear() {
      cache.clear();
   }
};

} // namespace lingodb::compiler::dialect::relalg

#endif // LINGODB_COMPILER_DIALECT_RELALG_AVAILABILITYCACHE_H
