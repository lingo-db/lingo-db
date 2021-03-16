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
::mlir::Type DBDialect::parseType(::mlir::DialectAsmParser& parser) const {
   StringRef memnonic;
   if (parser.parseKeyword(&memnonic)) {
      return Type();
   }
   auto loc=parser.getCurrentLocation();
   Type parsed= DBTypeParse(parser.getBuilder().getContext(), parser, memnonic);
   if(!parsed){
      parser.emitError(loc,"unknown type");
   }
   return parsed;
}

/// Print a type registered to this dialect.
void DBDialect::printType(::mlir::Type type,
                          ::mlir::DialectAsmPrinter& os) const {
   DBTypePrint(type, os);
}