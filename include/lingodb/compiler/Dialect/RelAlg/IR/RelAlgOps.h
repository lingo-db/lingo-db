#ifndef LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPS_H
#define LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSATypes.h"

#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsEnums.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h.inc"

#endif //LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPS_H
