#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.cpp.inc"
namespace mlir::subop {
void SubOperatorDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::subop