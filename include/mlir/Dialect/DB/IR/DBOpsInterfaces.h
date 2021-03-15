#ifndef DB_DBInterfaces
#define DB_DBInterfaces

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"


#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.h.inc"

#endif// DB_DBInterfaces
