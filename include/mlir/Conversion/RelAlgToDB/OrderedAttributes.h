#ifndef MLIR_CONVERSION_RELALGTODB_ORDEREDATTRIBUTES_H
#define MLIR_CONVERSION_RELALGTODB_ORDEREDATTRIBUTES_H
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace relalg {
class OrderedAttributes {
   std::vector<Type> types;
   std::vector<const mlir::relalg::Column*> attrs;

   public:
   static OrderedAttributes fromRefArr(ArrayAttr arrayAttr) {
      OrderedAttributes res;
      for (auto attr : arrayAttr) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>()) {
            res.insert(&attrRef.getColumn());
         }
         if (auto attrDef = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>()) {
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
   static OrderedAttributes fromVec(std::vector<const mlir::relalg::Column*> vec) {
      OrderedAttributes res;
      for (auto* attr : vec) {
         res.insert(attr);
      }
      return res;
   }
   mlir::Value resolve(TranslatorContext& context, size_t pos) {
      return context.getValueForAttribute(attrs[pos]);
   }
   mlir::Value pack(TranslatorContext& context, OpBuilder& builder, Location loc, std::vector<Value> additional = {}) {
      std::vector<Value> values(additional);
      for (size_t i = 0; i < attrs.size(); i++) {
         values.push_back(resolve(context, i));
      }
      if (values.size() == 0) {
         return builder.create<mlir::util::UndefOp>(loc, mlir::TupleType::get(builder.getContext()));
      }
      return builder.create<mlir::util::PackOp>(loc, values);
   }

   size_t insert(const mlir::relalg::Column* attr, Type alternativeType = {}) {
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
   void setValuesForColumns(TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope, ValueRange values) {
      for (size_t i = 0; i < attrs.size(); i++) {
         if (attrs[i]) {
            context.setValueForAttribute(scope, attrs[i], values[i]);
         }
      }
   }
   const std::vector<const mlir::relalg::Column*>& getAttrs() const {
      return attrs;
   }
   size_t getPos(const mlir::relalg::Column* attr) {
      return std::find(attrs.begin(), attrs.end(), attr) - attrs.begin();
   }
};
} // end namespace relalg
} // end namespace mlir
#endif // MLIR_CONVERSION_RELALGTODB_ORDEREDATTRIBUTES_H
