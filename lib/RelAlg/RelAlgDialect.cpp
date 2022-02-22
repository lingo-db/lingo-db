#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::relalg;

struct RelalgInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, BlockAndValueMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                                BlockAndValueMapping& valueMapping) const override {
      return true;
   }
};
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"
void RelAlgDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
      >();
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"
      >();
   addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"
      >();
   addInterfaces<RelalgInlinerInterface>();
   relationalAttributeManager.setContext(getContext());
}

::mlir::Attribute mlir::relalg::TableMetaDataAttr::parse(::mlir::AsmParser& parser, ::mlir::Type type) {
   StringAttr attr;
   if (parser.parseLess() || parser.parseAttribute(attr) || parser.parseGreater()) return Attribute();
   return mlir::relalg::TableMetaDataAttr::get(parser.getContext(), runtime::TableMetaData::deserialize(attr.str()));
}
void mlir::relalg::TableMetaDataAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getMeta()->serialize() << ">";
}
void mlir::relalg::RelationalAttributeDefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName() << ">";
}
::mlir::Attribute mlir::relalg::RelationalAttributeDefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   std::string str;
   if (parser.parseLess() || parser.parseString(&str) || parser.parseGreater()) return Attribute();
   return mlir::relalg::RelationalAttributeDefAttr::get(parser.getContext(), str, {}, {});
}
void mlir::relalg::RelationalAttributeRefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName() << ">";
}
::mlir::Attribute mlir::relalg::RelationalAttributeRefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseGreater()) return Attribute();
   return mlir::relalg::RelationalAttributeRefAttr::get(parser.getContext(), sym, {});
}
void mlir::relalg::SortSpecificationAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<"
           << ">";
}
::mlir::Attribute mlir::relalg::SortSpecificationAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   return mlir::relalg::SortSpecificationAttr::get(parser.getContext(), {}, {});
}
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc"
