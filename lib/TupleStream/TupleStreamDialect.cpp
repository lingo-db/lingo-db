#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"

#include "llvm/ADT/TypeSwitch.h"
using namespace mlir::tuples;

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
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.cpp.inc"
void TupleStreamDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/TupleStream/TupleStreamOps.cpp.inc"
      >();
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/TupleStream/TupleStreamOpsTypes.cpp.inc"
      >();
   addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.cpp.inc"
      >();
   addInterfaces<TupleStreamInlinerInterface>();
   columnManager.setContext(getContext());
   getContext()->loadDialect<mlir::db::DBDialect>();
   getContext()->loadDialect<mlir::dsa::DSADialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
}
void mlir::tuples::ColumnDefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName() << "," << getColumn().type;
   if (auto fromexisting = getFromExisting()) {
      printer << "," << fromexisting;
   }
   printer << ">";
}
::mlir::Attribute mlir::tuples::ColumnDefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
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
   auto columnDef = parser.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(sym, fromExisting);
   columnDef.getColumn().type = t;
   return columnDef;
}
void mlir::tuples::ColumnRefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName() << ">";
}
::mlir::Attribute mlir::tuples::ColumnRefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseGreater()) return Attribute();
   return parser.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(sym);
}

#include "mlir/Dialect/TupleStream/TupleStreamOpsDialect.cpp.inc"
