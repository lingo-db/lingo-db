#ifndef MLIR_DIALECT_DB_IR_DBOPS_H
#define MLIR_DIALECT_DB_IR_DBOPS_H


#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DB/IR/DBCollectionType.h"
#include "mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/DB/IR/DBOpsInterfaces.h"


#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.h.inc"
mlir::Type constructNullableBool(mlir::MLIRContext* context, mlir::ValueRange operands);
mlir::Type getBaseType(mlir::Type t);
#endif // MLIR_DIALECT_DB_IR_DBOPS_H
