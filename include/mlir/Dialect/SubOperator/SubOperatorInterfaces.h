#ifndef MLIR_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
#define MLIR_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
#include "mlir/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnFolding.h"
#include "mlir/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <string>
#include <vector>
#define GET_OP_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOpsInterfaces.h.inc"

#endif //MLIR_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
