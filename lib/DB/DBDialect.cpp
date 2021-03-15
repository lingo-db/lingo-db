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
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
      >();
}

/// Parse a type registered to this dialect.
::mlir::Type DBDialect::parseType(::mlir::DialectAsmParser &parser) const {
  if(!parser.parseOptionalKeyword("bool")){
    return mlir::db::BoolType::get(parser.getBuilder().getContext());
  }
  if(!parser.parseOptionalKeyword("int")){
    return mlir::db::IntType::get(parser.getBuilder().getContext());
  }
  return mlir::Type();
}

/// Print a type registered to this dialect.
void DBDialect::printType(::mlir::Type type,
                          ::mlir::DialectAsmPrinter &os) const {
    if(type.isa<mlir::db::BoolType>()){
      os<<"bool";
    }else if(type.isa<mlir::db::IntType>()){
      os<<"int";
    }

}