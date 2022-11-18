#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

mlir::Type mlir::dsa::RecordBatchType::getElementType() {
   return mlir::dsa::RecordType::get(getContext(), getRowType());
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
namespace mlir::dsa {
void DSADialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::dsa
