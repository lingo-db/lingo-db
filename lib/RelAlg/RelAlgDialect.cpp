
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include <mlir/Transforms/InliningUtils.h>

#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::relalg;

struct RelalgInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;

   //===--------------------------------------------------------------------===//
   // Analysis Hooks
   //===--------------------------------------------------------------------===//

   /// All call operations within toy can be inlined.
   bool isLegalToInline(Operation* call, Operation* callable,
                        bool wouldBeCloned) const final override{
      return true;
   }

   /// All operations within toy can be inlined.
   bool isLegalToInline(Operation*, Region*, bool,
                        BlockAndValueMapping&) const final override{
      return true;
   }
   virtual bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                                BlockAndValueMapping &valueMapping) const override{
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
   addAttributes<mlir::relalg::RelationalAttributeDefAttr>();
   addAttributes<mlir::relalg::RelationalAttributeRefAttr>();
   addAttributes<mlir::relalg::SortSpecificationAttr>();
   relationalAttributeManager.setContext(getContext());

}

::mlir::Attribute mlir::relalg::TableMetaDataAttr::parse(::mlir::AsmParser& parser, ::mlir::Type type) {
   if(parser.parseLess()) return Attribute();
   StringAttr attr;
   if(parser.parseAttribute(attr))return Attribute();
   if(parser.parseGreater())
      return Attribute();
   return mlir::relalg::TableMetaDataAttr::get(parser.getContext(), runtime::TableMetaData::deserialize(attr.str()));
}
void mlir::relalg::TableMetaDataAttr::print(::mlir::AsmPrinter& printer) const {
   printer<<"<"<<getMeta()->serialize()<<">";
}
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc"
