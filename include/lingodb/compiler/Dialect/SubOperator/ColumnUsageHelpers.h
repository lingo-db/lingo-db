#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_COLUMNUSAGEHELPERS_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_COLUMNUSAGEHELPERS_H

#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Operation.h"

namespace lingodb::compiler::dialect::subop {

/// Default fallback used by SubOperator::addUsedColumns. Walks every
/// attribute attached to `op` and inserts every tuples::Column reached
/// via a ColumnRefAttr (directly, transitively through ColumnDefAttr's
/// fromExisting chain, or through the SubOp ColumnRef/DefMemberMapping
/// attrs). Ops that move a column-bearing argument from an attribute
/// to a property must override addUsedColumns to also include that
/// property's columns - the default only sees what is still in
/// op->getAttrs().
void collectUsedColumnsFromAttrs(mlir::Operation* op,
                                 llvm::DenseSet<tuples::Column*>& out);

/// Default fallback used by SubOperator::addCreatedColumns. Walks every
/// attribute attached to `op` and inserts every tuples::Column defined
/// by a ColumnDefAttr (directly or via SubOp ColumnDefMemberMapping).
void collectCreatedColumnsFromAttrs(mlir::Operation* op,
                                    llvm::DenseSet<tuples::Column*>& out);

} // namespace lingodb::compiler::dialect::subop

#endif // LINGODB_COMPILER_DIALECT_SUBOPERATOR_COLUMNUSAGEHELPERS_H
