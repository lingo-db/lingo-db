#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/IR/DialectImplementation.h"
using namespace mlir;
using namespace mlir::db;

void DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
      >();
   registerTypes();
}