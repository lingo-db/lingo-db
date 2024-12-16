#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

struct TupleStreamInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(mlir::Region* dest, mlir::Region* src, bool wouldBeCloned,
                                mlir::IRMapping& valueMapping) const override {
      return true;
   }
};

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.cpp.inc"
void lingodb::compiler::dialect::tuples::TupleStreamDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.cpp.inc"

      >();
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.cpp.inc"

      >();
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.cpp.inc"

      >();
   addInterfaces<TupleStreamInlinerInterface>();
   columnManager.setContext(getContext());
   getContext()->loadDialect<db::DBDialect>();
   getContext()->loadDialect<dsa::DSADialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
}
void lingodb::compiler::dialect::tuples::ColumnDefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName() << "," << getColumn().type;
   if (auto fromexisting = getFromExisting()) {
      printer << "," << fromexisting;
   }
   printer << ">";
}
::mlir::Attribute lingodb::compiler::dialect::tuples::ColumnDefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   mlir::Type t;
   mlir::ArrayAttr fromExisting;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseComma() || parser.parseType(t)) return Attribute();
   if (parser.parseOptionalComma().succeeded()) {
      if (parser.parseAttribute(fromExisting)) {
         return Attribute();
      }
   }
   if (parser.parseGreater()) return Attribute();
   auto columnDef = parser.getContext()->getLoadedDialect<TupleStreamDialect>()->getColumnManager().createDef(sym, fromExisting);
   columnDef.getColumn().type = t;
   return columnDef;
}
void lingodb::compiler::dialect::tuples::ColumnRefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName() << ">";
}
::mlir::Attribute lingodb::compiler::dialect::tuples::ColumnRefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseGreater()) return Attribute();
   return parser.getContext()->getLoadedDialect<TupleStreamDialect>()->getColumnManager().createRef(sym);
}

#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsDialect.cpp.inc"
