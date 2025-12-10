#ifndef LINGODB_COMPILER_DIALECT_DB_IR_DBOPS_H
#define LINGODB_COMPILER_DIALECT_DB_IR_DBOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOpsEnums.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOpsInterfaces.h"
#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
struct ListMemory
    : public mlir::SideEffects::Resource::Base<ListMemory> {
   mlir::StringRef getName() final { return "ListMemory"; }
};
#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h.inc"
mlir::Type getBaseType(mlir::Type t);
mlir::Type wrapNullableType(mlir::MLIRContext* context, mlir::Type type, mlir::ValueRange values);
bool isIntegerType(mlir::Type, unsigned int width);
int getIntegerWidth(mlir::Type, bool isUnSigned);
#endif //LINGODB_COMPILER_DIALECT_DB_IR_DBOPS_H
