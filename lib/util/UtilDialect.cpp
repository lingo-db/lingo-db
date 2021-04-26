#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/DialectImplementation.h"
using namespace mlir;
using namespace mlir::util;

void UtilDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/util/UtilOps.cpp.inc"
      >();
}