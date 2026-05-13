#ifndef LINGODB_COMPILER_DIALECT_DB_IR_DBTYPES_H
#define LINGODB_COMPILER_DIALECT_DB_IR_DBTYPES_H

#include "lingodb/compiler/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace lingodb::compiler::dialect::db {
// Default ManagedType emitters. Declared here so the inline default bodies of
// the type interface compile against just `DBTypes.h`; the implementations
// live in `DBOps.cpp` where the `db.memory.*` op classes are fully defined.
void emitDefaultAddUse(::mlir::OpBuilder& builder, ::mlir::Location loc,
                       ::mlir::Value value);
void emitDefaultCleanupUse(::mlir::OpBuilder& builder, ::mlir::Location loc,
                           ::mlir::Value value, ::mlir::SymbolRefAttr elementFn);
::mlir::Value emitDefaultPromoteToGlobal(::mlir::OpBuilder& builder,
                                         ::mlir::Location loc,
                                         ::mlir::Value value);
} // namespace lingodb::compiler::dialect::db

#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypes.h.inc"

#endif //LINGODB_COMPILER_DIALECT_DB_IR_DBTYPES_H
