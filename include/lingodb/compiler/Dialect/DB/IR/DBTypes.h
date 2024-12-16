#ifndef LINGODB_COMPILER_DIALECT_DB_IR_DBTYPES_H
#define LINGODB_COMPILER_DIALECT_DB_IR_DBTYPES_H

#include "lingodb/compiler/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypes.h.inc"

#endif //LINGODB_COMPILER_DIALECT_DB_IR_DBTYPES_H
