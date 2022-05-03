#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"

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
struct ArithCmpICmpInterface
   : public CmpOpInterface::ExternalModel<ArithCmpICmpInterface, mlir::arith::CmpIOp> {
   // No need to define `exampleInterfaceHook` that has a default implementation
   // in `ExternalModel`. But it can be overridden if desired.
   bool isEqualityPred(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpIPredicate::eq;
   }
   bool isLessPred(mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpIPredicate::sle:
         case mlir::arith::CmpIPredicate::ule:
            return eq;
         case mlir::arith::CmpIPredicate::ult:
         case mlir::arith::CmpIPredicate::slt:
            return !eq;
         default: return false;
      }
   }
   bool isGreaterPred(mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpIPredicate::sge:
         case mlir::arith::CmpIPredicate::uge:
            return eq;
         case mlir::arith::CmpIPredicate::ugt:
         case mlir::arith::CmpIPredicate::sgt:
            return !eq;
         default: return false;
      }
   }
   mlir::Value getLeft(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getLhs();
   }
   mlir::Value getRight(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getRhs();
   }
};
struct ArithCmpFCmpInterface
   : public CmpOpInterface::ExternalModel<ArithCmpFCmpInterface, mlir::arith::CmpFOp> {
   // No need to define `exampleInterfaceHook` that has a default implementation
   // in `ExternalModel`. But it can be overridden if desired.
   bool isEqualityPred(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpFPredicate::OEQ || cmpOp.getPredicate() == mlir::arith::CmpFPredicate::UEQ;
   }
   bool isLessPred(mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpFPredicate::ULE:
         case mlir::arith::CmpFPredicate::OLE:
            return eq;
         case mlir::arith::CmpFPredicate::ULT:
         case mlir::arith::CmpFPredicate::OLT:
            return !eq;
         default: return false;
      }
   }
   bool isGreaterPred(mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpFPredicate::UGE:
         case mlir::arith::CmpFPredicate::OGE:
            return eq;
         case mlir::arith::CmpFPredicate::UGT:
         case mlir::arith::CmpFPredicate::OGT:
            return !eq;
         default: return false;
      }
   }
   mlir::Value getLeft(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getLhs();
   }
   mlir::Value getRight(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getRhs();
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
   columnManager.setContext(getContext());
   getContext()->loadDialect<mlir::db::DBDialect>();
   getContext()->loadDialect<mlir::dsa::DSADialect>();
   getContext()->loadDialect<mlir::arith::ArithmeticDialect>();
   mlir::arith::CmpIOp::attachInterface<ArithCmpFCmpInterface>(*getContext());
}

::mlir::Attribute mlir::relalg::TableMetaDataAttr::parse(::mlir::AsmParser& parser, ::mlir::Type type) {
   StringAttr attr;
   if (parser.parseLess() || parser.parseAttribute(attr) || parser.parseGreater()) return Attribute();
   return mlir::relalg::TableMetaDataAttr::get(parser.getContext(), runtime::TableMetaData::deserialize(attr.str()));
}
void mlir::relalg::TableMetaDataAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<";
   printer.printAttribute(StringAttr::get(getContext(), getMeta()->serialize()));
   printer << ">";
}
void mlir::relalg::ColumnDefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName()<<","<<getColumn().type;
   if (auto fromexisting = getFromExisting()) {
      printer << "," << fromexisting;
   }
   printer << ">";
}
::mlir::Attribute mlir::relalg::ColumnDefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   mlir::Type t;
   mlir::ArrayAttr fromExisting;
   if (parser.parseLess() || parser.parseAttribute(sym)||parser.parseComma()||parser.parseType(t)) return Attribute();
   if (parser.parseOptionalComma().succeeded()) {
      if (parser.parseAttribute(fromExisting)) {
         return Attribute();
      }
   }
   if (parser.parseGreater()) return Attribute();
   auto columnDef= parser.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager().createDef(sym, fromExisting);
   columnDef.getColumn().type=t;
   return columnDef;
}
void mlir::relalg::ColumnRefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName() << ">";
}
::mlir::Attribute mlir::relalg::ColumnRefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseGreater()) return Attribute();
   return parser.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager().createRef(sym);
}
void mlir::relalg::SortSpecificationAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getAttr().getName() << "," << stringifyEnum(getSortSpec()) << ">";
}
::mlir::Attribute mlir::relalg::SortSpecificationAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   std::string sortSpecDescr;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseComma() || parser.parseKeywordOrString(&sortSpecDescr) || parser.parseGreater()) {
      return mlir::Attribute();
   }
   auto sortSpec = symbolizeSortSpec(sortSpecDescr);
   if (!sortSpec.hasValue()) {
      return {};
   }
   auto columnRefAttr = parser.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager().createRef(sym);
   return mlir::relalg::SortSpecificationAttr::get(parser.getContext(), columnRefAttr, sortSpec.getValue());
}
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc"
