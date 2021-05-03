#ifndef MLIR_DIALECT_DB_IR_DBTYPES_H
#define MLIR_DIALECT_DB_IR_DBTYPES_H

#include "mlir/Dialect/DB/IR/DBCollectionType.h"
#include "mlir/Dialect/DB/IR/DBType.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsTypes.h.inc"

#endif // MLIR_DIALECT_DB_IR_DBTYPES_H
