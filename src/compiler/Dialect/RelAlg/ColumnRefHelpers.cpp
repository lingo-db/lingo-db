#include "lingodb/compiler/Dialect/RelAlg/ColumnRefHelpers.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"

namespace lingodb::compiler::dialect::relalg {
namespace {

mlir::Attribute updateAttribute(mlir::Attribute attr, ColumnRefReplaceFn fn) {
   if (auto refAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
      return fn(refAttr);
   }
   if (auto defAttr = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
      if (auto fromExisting = defAttr.getFromExisting()) {
         return tuples::ColumnDefAttr::get(attr.getContext(), defAttr.getName(),
                                           defAttr.getColumnPtr(),
                                           updateAttribute(fromExisting, fn));
      }
      return attr;
   }
   if (auto sortSpec = mlir::dyn_cast<relalg::SortSpecificationAttr>(attr)) {
      return relalg::SortSpecificationAttr::get(attr.getContext(),
                                                fn(sortSpec.getAttr()),
                                                sortSpec.getSortSpec());
   }
   if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      llvm::SmallVector<mlir::Attribute> newElems;
      newElems.reserve(arrayAttr.size());
      for (auto elem : arrayAttr) {
         newElems.push_back(updateAttribute(elem, fn));
      }
      return mlir::ArrayAttr::get(attr.getContext(), newElems);
   }
   return attr;
}

} // namespace

void replaceColumnRefsInAttrs(mlir::Operation* op, ColumnRefReplaceFn fn) {
   for (auto namedAttr : op->getAttrs()) {
      op->setAttr(namedAttr.getName(),
                  updateAttribute(namedAttr.getValue(), fn));
   }
   // Property-backed column refs are not visible via getAttrs(); rewrite
   // them explicitly for the ops we know about.
   if (auto getColumnOp = mlir::dyn_cast<tuples::GetColumnOp>(op)) {
      if (auto* col = getColumnOp.getAttr()) {
         auto& colManager = op->getContext()
                               ->getLoadedDialect<tuples::TupleStreamDialect>()
                               ->getColumnManager();
         auto newRef = fn(colManager.createRef(col));
         getColumnOp.setAttr(&newRef.getColumn());
      }
   }
}

} // namespace lingodb::compiler::dialect::relalg
