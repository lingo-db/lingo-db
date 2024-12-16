#ifndef LINGODB_COMPILER_DIALECT_DSA_IR_DSAOPS_H
#define LINGODB_COMPILER_DIALECT_DSA_IR_DSAOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lingodb/compiler/Dialect/DSA/IR/DSAOpsEnums.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSAOpsInterfaces.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.h.inc"
mlir::Type getBaseType(mlir::Type t);
bool isIntegerType(mlir::Type, unsigned int width);
int getIntegerWidth(mlir::Type, bool isUnSigned);
#endif //LINGODB_COMPILER_DIALECT_DSA_IR_DSAOPS_H
