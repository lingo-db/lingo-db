#ifndef MLIR_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
#define MLIR_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H

#include "mlir/Dialect/TupleStream/Column.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/TupleStream/TupleStreamOps.h.inc"

#endif // MLIR_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
