
#ifndef MLIR_DIALECT_RELALG_ATTRIBUTES_H
#define MLIR_DIALECT_RELALG_ATTRIBUTES_H
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <mlir/Dialect/RelAlg/IR/RelationalAttribute.h>
namespace mlir::relalg {
class Attributes {
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;
   attribute_set attributes;

   public:
   Attributes intersect(const Attributes& other) const {
      Attributes result;
      for (auto *x : attributes) {
         if (other.attributes.contains(x)) {
            result.insert(x);
         }
      }
      return result;
   }
   bool empty() const {
      return attributes.empty();
   }
   void insert(mlir::relalg::RelationalAttribute* attr) {
      attributes.insert(attr);
   }
   Attributes& insert(const Attributes& other) {
      attributes.insert(other.attributes.begin(), other.attributes.end());
      return *this;
   }
   void remove(const Attributes& other) {
      for (auto *elem : other.attributes) {
         attributes.erase(elem);
      }
   }
   bool intersects(const Attributes& others) const {
      for (auto* x : attributes) {
         if (others.attributes.contains(x)) {
            return true;
         }
      }
      return false;
   }

   bool isSubsetOf(const Attributes& others) const {
      for (auto* x : attributes) {
         if (!others.attributes.contains(x)) {
            return false;
         }
      }
      return true;
   }
   [[nodiscard]] auto begin() const {
      return attributes.begin();
   }
   [[nodiscard]] auto end() const {
      return attributes.end();
   }
   void dump(MLIRContext* context) {
      auto& attributeManager = context->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      for (auto* x : attributes) {
         auto [scope, name] = attributeManager.getName(x);
         llvm::dbgs() << x << "(" << scope << "," << name << "),";
      }
   }
   ArrayAttr asRefArrayAttr(MLIRContext* context) {
      auto& attributeManager = context->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();

      std::vector<Attribute> refAttrs;
      for (auto* attr : attributes) {
         refAttrs.push_back(attributeManager.createRef(attr));
      }
      return ArrayAttr::get(context, refAttrs);
   }
   static Attributes fromArrayAttr(ArrayAttr arrayAttr) {
      Attributes res;
      for (auto attr : arrayAttr) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            res.insert(&attrRef.getRelationalAttribute());
         } else if (auto attrDef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>()) {
            res.insert(&attrDef.getRelationalAttribute());
         }
      }
      return res;
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_ATTRIBUTES_H
