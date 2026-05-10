#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STEPGRAPHUTILS_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STEPGRAPHUTILS_H
#include "mlir/IR/Operation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace lingodb::compiler::dialect::subop {

/// Kahn's topological sort with deterministic tie-breaking.
///
/// `nodes` are the items to order. `deps[a]` is the set of nodes `a` depends
/// on (so `a` is emitted *after* every entry in `deps[a]`). When multiple
/// nodes are simultaneously ready, the one with smallest `priority[n]` is
/// emitted first; this is the standard "first walk position" trick used by
/// both Organize/SplitIntoNested execution-step passes.
///
/// Returns the ordered list. If the graph has a cycle, the returned vector
/// has size < `nodes.size()` — the caller should diagnose that condition.
std::vector<mlir::Operation*> kahnTopoSort(
   llvm::ArrayRef<mlir::Operation*> nodes,
   const llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>>& deps,
   const llvm::DenseMap<mlir::Operation*, size_t>& priority);

} // namespace lingodb::compiler::dialect::subop
#endif // LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STEPGRAPHUTILS_H
