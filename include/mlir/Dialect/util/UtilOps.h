#ifndef UTIL_UTILOPS_H
#define UTIL_UTILOPS_H


#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"



#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/util/UtilOps.h.inc"

#endif// UTIL_UTILOPS_H
