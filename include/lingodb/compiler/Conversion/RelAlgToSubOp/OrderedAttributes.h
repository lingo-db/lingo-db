#ifndef LINGODB_COMPILER_CONVERSION_RELALGTOSUBOP_ORDEREDATTRIBUTES_H
#define LINGODB_COMPILER_CONVERSION_RELALGTOSUBOP_ORDEREDATTRIBUTES_H
#include "lingodb/compiler/Dialect/RelAlg/ColumnSet.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "mlir/IR/Value.h"

namespace lingodb::compiler::dialect {
namespace relalg {
class OrderedAttributes {
   std::vector<mlir::Type> types;
   std::vector<const tuples::Column*> attrs;

   public:
   static OrderedAttributes fromRefArr(mlir::ArrayAttr arrayAttr) {
      OrderedAttributes res;
      for (auto attr : arrayAttr) {
         if (auto attrRef = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(attr)) {
            res.insert(&attrRef.getColumn());
         }
         if (auto attrDef = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr)) {
            res.insert(&attrDef.getColumn());
         }
      }
      return res;
   }
   static OrderedAttributes fromColumns(relalg::ColumnSet attrs) {
      OrderedAttributes res;
      for (auto* attr : attrs) {
         res.insert(attr);
      }
      return res;
   }
   static OrderedAttributes fromVec(std::vector<const tuples::Column*> vec) {
      OrderedAttributes res;
      for (auto* attr : vec) {
         res.insert(attr);
      }
      return res;
   }

   size_t insert(const tuples::Column* attr, mlir::Type alternativeType = {}) {
      attrs.push_back(attr);
      if (attr) {
         types.push_back(attr->type);
      } else {
         types.push_back(alternativeType);
      }
      return attrs.size() - 1;
   }
   mlir::TupleType getTupleType(mlir::MLIRContext* ctxt, std::vector<mlir::Type> additional = {}) {
      std::vector<mlir::Type> tps(additional);
      tps.insert(tps.end(), types.begin(), types.end());
      return mlir::TupleType::get(ctxt, tps);
   }
   const std::vector<const tuples::Column*>& getAttrs() const {
      return attrs;
   }
   mlir::ArrayAttr getArrayAttr(mlir::MLIRContext* context) const {
      std::vector<mlir::Attribute> attrVec;
      auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      for (const auto* a : attrs) {
         attrVec.push_back(colManager.createRef(a));
      }
      return mlir::ArrayAttr::get(context, attrVec);
   }
   mlir::ArrayAttr getDefArrayAttr(mlir::MLIRContext* context) const {
      std::vector<mlir::Attribute> attrVec;
      auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      for (const auto* a : attrs) {
         attrVec.push_back(colManager.createDef(a));
      }
      return mlir::ArrayAttr::get(context, attrVec);
   }
   size_t getPos(const tuples::Column* attr) {
      return std::find(attrs.begin(), attrs.end(), attr) - attrs.begin();
   }
};
} // end namespace relalg
} // end namespace lingodb::compiler::dialect
#endif //LINGODB_COMPILER_CONVERSION_RELALGTOSUBOP_ORDEREDATTRIBUTES_H
