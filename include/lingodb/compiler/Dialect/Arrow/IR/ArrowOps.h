#ifndef LINGODB_COMPILER_DIALECT_ARROW_IR_ARROWOPS_H
#define LINGODB_COMPILER_DIALECT_ARROW_IR_ARROWOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOps.h.inc"
#endif //LINGODB_COMPILER_DIALECT_ARROW_IR_ARROWOPS_H
