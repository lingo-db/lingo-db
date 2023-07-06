#include "mlir/Dialect/cranelift/CraneliftDialect.h"
#include "mlir/Dialect/cranelift/CraneliftOps.h"
#include "mlir/IR/DialectImplementation.h"


void mlir::cranelift::CraneliftDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/cranelift/CraneliftOps.cpp.inc"
      >();
   registerTypes();
}
#include "mlir/Dialect/cranelift/CraneliftOpsDialect.cpp.inc"
