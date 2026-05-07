#ifndef LINGODB_COMPILER_DIALECT_RELALG_COLUMNREFHELPERS_H
#define LINGODB_COMPILER_DIALECT_RELALG_COLUMNREFHELPERS_H

#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/Operation.h"

namespace lingodb::compiler::dialect::relalg {

using ColumnRefReplaceFn = llvm::function_ref<tuples::ColumnRefAttr(tuples::ColumnRefAttr)>;

/// Recursively rewrite every ColumnRefAttr reachable via `op`'s attribute
/// dictionary, replacing each one with `fn(ref)`. Descends into
/// ColumnDefAttr.fromExisting, SortSpecificationAttr, and ArrayAttr.
/// Properties are NOT touched - callers that also need to rewrite refs
/// stored in properties must additionally invoke
/// Operator::replaceColumnRefs on ops that implement that interface.
void replaceColumnRefsInAttrs(mlir::Operation* op, ColumnRefReplaceFn fn);

} // namespace lingodb::compiler::dialect::relalg

#endif // LINGODB_COMPILER_DIALECT_RELALG_COLUMNREFHELPERS_H
