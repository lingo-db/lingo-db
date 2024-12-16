#ifndef LINGODB_COMPILER_DIALECT_UTIL_UTILOPS_H
#define LINGODB_COMPILER_DIALECT_UTIL_UTILOPS_H

#include "lingodb/compiler/Dialect/util/UtilTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/util/UtilOps.h.inc"

#endif //LINGODB_COMPILER_DIALECT_UTIL_UTILOPS_H
