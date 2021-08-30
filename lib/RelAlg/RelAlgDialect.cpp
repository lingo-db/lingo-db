#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include <mlir/Transforms/InliningUtils.h>

#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::relalg;

struct RelalgInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;

   //===--------------------------------------------------------------------===//
   // Analysis Hooks
   //===--------------------------------------------------------------------===//

   /// All call operations within toy can be inlined.
   bool isLegalToInline(Operation* call, Operation* callable,
                        bool wouldBeCloned) const final {
      return true;
   }

   /// All operations within toy can be inlined.
   bool isLegalToInline(Operation*, Region*, bool,
                        BlockAndValueMapping&) const final {
      return true;
   }
   virtual bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                                BlockAndValueMapping &valueMapping) const {
      return true;
   }
};
void RelAlgDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
      >();
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"
      >();
   addInterfaces<RelalgInlinerInterface>();
   addAttributes<mlir::relalg::RelationalAttributeDefAttr>();
   addAttributes<mlir::relalg::RelationalAttributeRefAttr>();
   addAttributes<mlir::relalg::SortSpecificationAttr>();
   relationalAttributeManager.setContext(getContext());

}

/// Parse a type registered to this dialect.
::mlir::Type RelAlgDialect::parseType(::mlir::DialectAsmParser& parser) const {
   if (!parser.parseOptionalKeyword("relation")) {
      return mlir::relalg::TupleStreamType::get(parser.getBuilder().getContext());
   }
   if (!parser.parseOptionalKeyword("tuple")) {
      return mlir::relalg::TupleType::get(parser.getBuilder().getContext());
   }
   return mlir::Type();
}

/// Print a type registered to this dialect.
void RelAlgDialect::printType(::mlir::Type type,
                              ::mlir::DialectAsmPrinter& os) const {
   if (type.isa<mlir::relalg::TupleStreamType>()) {
      os << "relation";
   }
   if (type.isa<mlir::relalg::TupleType>()) {
      os << "tuple";
   }
}
::mlir::Attribute
RelAlgDialect::parseAttribute(::mlir::DialectAsmParser& parser,
                              ::mlir::Type type) const {
   if (!parser.parseOptionalKeyword("attr_def")) {
      StringRef name;
      if (parser.parseLBrace() || parser.parseOptionalString(&name) || parser.parseRBrace())
         return mlir::Attribute();
      return parser.getBuilder().getContext()->getLoadedDialect<RelAlgDialect>()->getRelationalAttributeManager().createDef(parser.getBuilder().getSymbolRefAttr(name));
   }
   return mlir::Attribute();
}
void RelAlgDialect::printAttribute(::mlir::Attribute attr,
                                   ::mlir::DialectAsmPrinter& os) const {
   if (auto attrDef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>()) {
      os << "attr_def(\\\"" << attrDef.getName() << "\\\")";
   }
}
