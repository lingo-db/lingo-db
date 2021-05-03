#ifndef MLIR_DIALECT_DB_IR_DBOPSINTERFACES_H
#define MLIR_DIALECT_DB_IR_DBOPSINTERFACES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"


#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.h.inc"

#endif // MLIR_DIALECT_DB_IR_DBOPSINTERFACES_H
