#ifndef DB_DBTYPES_H
#define DB_DBTYPES_H

#include "mlir/Dialect/DB/IR/DBType.h"
#include "mlir/Dialect/DB/IR/DBCollectionType.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsTypes.h.inc"

#endif // DB_DBTYPES_H
