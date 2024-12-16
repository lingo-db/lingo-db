#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lingodb::compiler::dialect::relalg;

struct RelalgInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                                IRMapping& valueMapping) const override {
      return true;
   }
};
struct RelAlgFoldInterface : public DialectFoldInterface {
   using DialectFoldInterface::DialectFoldInterface;

   bool shouldMaterializeInto(Region* region) const final {
      return true;
   }
};
struct ArithCmpICmpInterface
   : public CmpOpInterface::ExternalModel<ArithCmpICmpInterface, mlir::arith::CmpIOp> {
   // No need to define `exampleInterfaceHook` that has a default implementation
   // in `ExternalModel`. But it can be overridden if desired.
   bool isEqualityPred(mlir::Operation* op, bool nullsAreEqual) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpIPredicate::eq;
   }
   bool isUnequalityPred(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpIPredicate::ne;
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
   bool isEqualityPred(mlir::Operation* op, bool nullsAreEqual) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpFPredicate::OEQ || cmpOp.getPredicate() == mlir::arith::CmpFPredicate::UEQ;
   }
   bool isUnequalityPred(mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpFPredicate::ONE || cmpOp.getPredicate() == mlir::arith::CmpFPredicate::UNE;
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
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"
void RelAlgDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"

      >();

   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"

      >();
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"

      >();
   addInterfaces<RelalgInlinerInterface>();
   addInterfaces<RelAlgFoldInterface>();
   getContext()->loadDialect<db::DBDialect>();
   getContext()->loadDialect<dsa::DSADialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
   getContext()->loadDialect<tuples::TupleStreamDialect>();

   mlir::arith::CmpIOp::attachInterface<ArithCmpICmpInterface>(*getContext());
}

::mlir::Attribute TableMetaDataAttr::parse(::mlir::AsmParser& parser, ::mlir::Type type) {
   StringAttr attr;
   if (parser.parseLess() || parser.parseAttribute(attr) || parser.parseGreater()) return Attribute();
   return TableMetaDataAttr::get(parser.getContext(), runtime::TableMetaData::deserialize(attr.str()));
}
void TableMetaDataAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<";
   printer.printAttribute(StringAttr::get(getContext(), getMeta()->serialize()));
   printer << ">";
}
void SortSpecificationAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getAttr().getName() << "," << stringifyEnum(getSortSpec()) << ">";
}
::mlir::Attribute SortSpecificationAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   std::string sortSpecDescr;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseComma() || parser.parseKeywordOrString(&sortSpecDescr) || parser.parseGreater()) {
      return mlir::Attribute();
   }
   auto sortSpec = symbolizeSortSpec(sortSpecDescr);
   if (!sortSpec.has_value()) {
      return {};
   }
   auto columnRefAttr = parser.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createRef(sym);
   return SortSpecificationAttr::get(parser.getContext(), columnRefAttr, sortSpec.value());
}
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc"
