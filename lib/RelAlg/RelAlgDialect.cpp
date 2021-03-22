#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"

#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::relalg;
void RelAlgDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
      >();
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"
      >();
   addAttributes<mlir::relalg::RelationalAttributeDefAttr>();
   addAttributes<mlir::relalg::RelationalAttributeRefAttr>();
   addAttributes<mlir::relalg::SortSpecificationAttr>();

}

/// Parse a type registered to this dialect.
::mlir::Type RelAlgDialect::parseType(::mlir::DialectAsmParser& parser) const {
   if (!parser.parseOptionalKeyword("relation")) {
      return mlir::relalg::RelationType::get(parser.getBuilder().getContext());
   }
   if (!parser.parseOptionalKeyword("tuple")) {
      return mlir::relalg::TupleType::get(parser.getBuilder().getContext());
   }
   return mlir::Type();
}

/// Print a type registered to this dialect.
void RelAlgDialect::printType(::mlir::Type type,
                              ::mlir::DialectAsmPrinter& os) const {
   if (type.isa<mlir::relalg::RelationType>()) {
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
      return mlir::relalg::RelationalAttributeDefAttr::get(
         parser.getBuilder().getContext(), name,
         std::make_shared<RelationalAttribute>(mlir::db::BoolType::get(parser.getBuilder().getContext(), false)),
            Attribute());
   }
   return mlir::Attribute();
}
void RelAlgDialect::printAttribute(::mlir::Attribute attr,
                                   ::mlir::DialectAsmPrinter& os) const {
   if (auto attr_def = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>()) {
      os << "attr_def(\\\"" << attr_def.getName() << "\\\")";
   }
}
