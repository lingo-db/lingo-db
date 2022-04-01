#ifndef MLIR_DIALECT_RELALG_IR_RELALGOPS_H
#define MLIR_DIALECT_RELALG_IR_RELALGOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"

#include "mlir/Dialect/RelAlg/IR/Column.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsEnums.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h.inc"

#endif // MLIR_DIALECT_RELALG_IR_RELALGOPS_H
