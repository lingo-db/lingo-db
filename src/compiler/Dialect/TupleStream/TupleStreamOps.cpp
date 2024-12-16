#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <queue>

using namespace mlir;

namespace {
using namespace lingodb::compiler::dialect;
tuples::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
}

ParseResult parseCustRef(OpAsmParser& parser, tuples::ColumnRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}

void printCustRef(OpAsmPrinter& p, mlir::Operation* op, tuples::ColumnRefAttr attr) {
   p << attr.getName();
}
} // namespace

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.cpp.inc"