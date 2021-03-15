#ifndef DB_DBOPS_H
#define DB_DBOPS_H


#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/DB/IR/DBOpsInterfaces.h"


#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.h.inc"

#endif// DB_DBOPS_H
