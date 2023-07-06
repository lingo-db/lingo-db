#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include <llvm/ADT/TypeSwitch.h>

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOpsAttributes.cpp.inc"

void mlir::subop::SubOperatorDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/SubOperator/SubOperatorOpsAttributes.cpp.inc"
      >();
}