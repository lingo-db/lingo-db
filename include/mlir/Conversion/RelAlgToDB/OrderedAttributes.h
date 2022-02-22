#ifndef DB_DIALECTS_ORDEREDATTRIBUTES_H
#define DB_DIALECTS_ORDEREDATTRIBUTES_H
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Dialect/RelAlg/Attributes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Value.h"

namespace mlir::relalg {
class OrderedAttributes {
   std::vector<Type> types;
   std::vector<const mlir::relalg::RelationalAttribute*> attrs;

   public:
   static OrderedAttributes fromRefArr(ArrayAttr arrayAttr) {
      OrderedAttributes res;
      for (auto attr : arrayAttr) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            res.insert(&attrRef.getRelationalAttribute());
         }
         if (auto attrDef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>()) {
            res.insert(&attrDef.getRelationalAttribute());
         }
      }
      return res;
   }
   static OrderedAttributes fromAttributes(mlir::relalg::Attributes attrs) {
      OrderedAttributes res;
      for (auto* attr : attrs) {
         res.insert(attr);
      }
      return res;
   }
   static OrderedAttributes fromVec(std::vector<const mlir::relalg::RelationalAttribute*> vec) {
      OrderedAttributes res;
      for (auto* attr : vec) {
         res.insert(attr);
      }
      return res;
   }
   mlir::Value pack(TranslatorContext& context, OpBuilder& builder, Location loc, std::vector<Value> additional = {}) {
      std::vector<Value> values(additional);
      for (const auto* attr : attrs) {
         values.push_back(context.getValueForAttribute(attr));
      }
      if (values.size() == 0) {
         return builder.create<mlir::util::UndefTupleOp>(loc, mlir::TupleType::get(builder.getContext()));
      }
      return builder.create<mlir::util::PackOp>(loc, values);
   }
   size_t insert(const mlir::relalg::RelationalAttribute* attr, Type alternativeType = {}) {
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
   void setValuesForAttributes(TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope, ValueRange values) {
      for (size_t i = 0; i < attrs.size(); i++) {
         if (attrs[i]) {
            context.setValueForAttribute(scope, attrs[i], values[i]);
         }
      }
   }
   const std::vector<const mlir::relalg::RelationalAttribute*>& getAttrs() const {
      return attrs;
   }
   size_t getPos(const mlir::relalg::RelationalAttribute* attr) {
      return std::find(attrs.begin(), attrs.end(), attr) - attrs.begin();
   }
};
}

#endif //DB_DIALECTS_ORDEREDATTRIBUTES_H
