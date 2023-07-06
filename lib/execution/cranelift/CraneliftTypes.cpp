#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/cranelift/CraneliftDialect.h"
#include "mlir/Dialect/cranelift/CraneliftOps.h"
#include "mlir/Dialect/cranelift/CraneliftTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include <unordered_set>

using namespace mlir;



#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/cranelift/CraneliftOpsTypes.cpp.inc"

namespace mlir::cranelift {
void CraneliftDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/cranelift/CraneliftOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::cranelift
