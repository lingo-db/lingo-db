#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPSTYPES_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPSTYPES_H

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypes.h.inc"

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPSTYPES_H
