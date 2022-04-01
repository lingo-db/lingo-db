#ifndef MLIR_DIALECT_UTIL_UTILOPS_H
#define MLIR_DIALECT_UTIL_UTILOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/util/UtilOps.h.inc"

#endif // MLIR_DIALECT_UTIL_UTILOPS_H
