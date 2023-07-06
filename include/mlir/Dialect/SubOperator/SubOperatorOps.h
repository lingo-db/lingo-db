#ifndef MLIR_DIALECT_SUBOPERATOR_SUBOPERATOROPS_H
#define MLIR_DIALECT_SUBOPERATOR_SUBOPERATOROPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"

#include "mlir/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.h"

#include "mlir/Dialect/TupleStream/Column.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "mlir/Dialect/SubOperator/SubOperatorOpsEnums.h.inc"
#include "mlir/IR/Builders.h"
#define GET_OP_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOps.h.inc"

#endif // MLIR_DIALECT_SUBOPERATOR_SUBOPERATOROPS_H
