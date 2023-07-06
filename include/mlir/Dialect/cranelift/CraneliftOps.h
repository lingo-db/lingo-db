#ifndef MLIR_DIALECT_CRANELIFT_CRANELIFTOPS_H
#define MLIR_DIALECT_CRANELIFT_CRANELIFTOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/cranelift/CraneliftEnums.h"
#include "mlir/Dialect/cranelift/CraneliftTypes.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/cranelift/CraneliftOps.h.inc"

#endif // MLIR_DIALECT_CRANELIFT_CRANELIFTOPS_H
