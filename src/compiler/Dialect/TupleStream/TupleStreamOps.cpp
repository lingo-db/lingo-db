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

//===--------------------------------------------------------------------===//
// Property hooks for tuples::Column* (used by tuples.getcol). The Column is
// owned by the dialect-scoped ColumnManager, so the raw pointer is stable for
// the lifetime of the IR. Generic-form bridging goes via ColumnRefAttr to
// match the textual form.
//===--------------------------------------------------------------------===//
mlir::Attribute convertToAttribute(MLIRContext* ctx, tuples::Column* col) {
   if (!col) return {};
   auto& colManager = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   return colManager.createRef(col);
}
mlir::LogicalResult convertFromAttribute(tuples::Column*& col, mlir::Attribute attr,
                                         std::function<mlir::InFlightDiagnostic()> emitError) {
   if (!attr) {
      col = nullptr;
      return mlir::success();
   }
   if (auto refAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
      col = &refAttr.getColumn();
      return mlir::success();
   }
   if (auto defAttr = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
      col = &defAttr.getColumn();
      return mlir::success();
   }
   return emitError() << "expected tuples::ColumnRefAttr or ColumnDefAttr for Column* property";
}
void writeToMlirBytecode(mlir::DialectBytecodeWriter& writer, tuples::Column* col) {
   //TODO: bytecode round-trip for Column* (write scope+name, resolve via
   //      ColumnManager on read). Not exercised by current pipeline.
}
mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader& reader, tuples::Column*& col) {
   //TODO: see writeToMlirBytecode.
   return mlir::failure();
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

// Overloads of parseCustRef / printCustRef for tuples::Column* properties.
// Reuses the ColumnRefAttr text format and the ColumnManager that owns the
// columns; the lossless ColumnRefAttr <-> Column* conversion keeps the
// custom assembly format identical to the pre-migration form.
ParseResult parseCustRef(OpAsmParser& parser, tuples::Column*& col) {
   tuples::ColumnRefAttr attr;
   if (parseCustRef(parser, attr)) return failure();
   col = &attr.getColumn();
   return success();
}
void printCustRef(OpAsmPrinter& p, mlir::Operation* op, tuples::Column* col) {
   auto& colManager = op->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   printCustRef(p, op, colManager.createRef(col));
}
} // namespace

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.cpp.inc"