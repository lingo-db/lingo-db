#ifndef MLIR_DIALECT_SUBOPERATOR_SUBOPERATOROPSTYPES_H
#define MLIR_DIALECT_SUBOPERATOR_SUBOPERATOROPSTYPES_H

#include "mlir/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/SubOperator/SubOperatorOpsTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.h.inc"

#endif // MLIR_DIALECT_SUBOPERATOR_SUBOPERATOROPSTYPES_H
