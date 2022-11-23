#ifndef MLIR_CONVERSION_RELALGTOSUBOP_ORDEREDATTRIBUTES_H
#define MLIR_CONVERSION_RELALGTOSUBOP_ORDEREDATTRIBUTES_H
#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace relalg {
class OrderedAttributes {
   std::vector<Type> types;
   std::vector<const mlir::tuples::Column*> attrs;

   public:
   static OrderedAttributes fromRefArr(ArrayAttr arrayAttr) {
      OrderedAttributes res;
      for (auto attr : arrayAttr) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>()) {
            res.insert(&attrRef.getColumn());
         }
         if (auto attrDef = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>()) {
            res.insert(&attrDef.getColumn());
         }
      }
      return res;
   }
   static OrderedAttributes fromColumns(mlir::relalg::ColumnSet attrs) {
      OrderedAttributes res;
      for (auto* attr : attrs) {
         res.insert(attr);
      }
      return res;
   }
   static OrderedAttributes fromVec(std::vector<const mlir::tuples::Column*> vec) {
      OrderedAttributes res;
      for (auto* attr : vec) {
         res.insert(attr);
      }
      return res;
   }

   size_t insert(const mlir::tuples::Column* attr, Type alternativeType = {}) {
      attrs.push_back(attr);
      if (attr) {
         types.push_back(attr->type);
      } else {
         types.push_back(alternativeType);
      }
      return attrs.size() - 1;
   }
   mlir::TupleType getTupleType(mlir::MLIRContext* ctxt, std::vector<Type> additional = {}) {
      std::vector<Type> tps(additional);
      tps.insert(tps.end(), types.begin(), types.end());
      return mlir::TupleType::get(ctxt, tps);
   }
   const std::vector<const mlir::tuples::Column*>& getAttrs() const {
      return attrs;
   }
   mlir::ArrayAttr getArrayAttr(mlir::MLIRContext* context) const {
      std::vector<mlir::Attribute> attrVec;
      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      for (const auto* a : attrs) {
         attrVec.push_back(colManager.createRef(a));
      }
      return mlir::ArrayAttr::get(context, attrVec);
   }
   mlir::ArrayAttr getDefArrayAttr(mlir::MLIRContext* context) const {
      std::vector<mlir::Attribute> attrVec;
      auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      for (const auto* a : attrs) {
         attrVec.push_back(colManager.createDef(a));
      }
      return mlir::ArrayAttr::get(context, attrVec);
   }
   size_t getPos(const mlir::tuples::Column* attr) {
      return std::find(attrs.begin(), attrs.end(), attr) - attrs.begin();
   }
};
} // end namespace relalg
} // end namespace mlir
#endif // MLIR_CONVERSION_RELALGTOSUBOP_ORDEREDATTRIBUTES_H
