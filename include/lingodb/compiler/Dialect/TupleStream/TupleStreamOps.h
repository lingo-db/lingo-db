#ifndef LINGODB_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
#define LINGODB_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H

#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h.inc"

#endif //LINGODB_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMOPS_H
