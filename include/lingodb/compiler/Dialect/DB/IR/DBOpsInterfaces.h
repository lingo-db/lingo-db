#ifndef LINGODB_COMPILER_DIALECT_DB_IR_DBOPSINTERFACES_H
#define LINGODB_COMPILER_DIALECT_DB_IR_DBOPSINTERFACES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/DB/IR/DBOpsInterfaces.h.inc"

#endif //LINGODB_COMPILER_DIALECT_DB_IR_DBOPSINTERFACES_H
