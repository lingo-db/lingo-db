#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPS_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPS_H

#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsEnums.h.inc"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypes.h"
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h.inc"

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPS_H
