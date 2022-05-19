
#ifndef MLIR_DIALECT_RELALG_COLUMNSET_H
#define MLIR_DIALECT_RELALG_COLUMNSET_H
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/RelAlg/IR/Column.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>
namespace mlir::relalg {
class ColumnSet {
   using attribute_set = llvm::SmallPtrSet<const mlir::relalg::Column*, 8>;
   attribute_set attributes;

   public:
   ColumnSet intersect(const ColumnSet& other) const {
      ColumnSet result;
      for (const auto* x : attributes) {
         if (other.attributes.contains(x)) {
            result.insert(x);
         }
      }
      return result;
   }
   bool empty() const {
      return attributes.empty();
   }
   size_t size() const {
      return attributes.size();
   }
   void insert(const mlir::relalg::Column* attr) {
      attributes.insert(attr);
   }
   bool contains(const mlir::relalg::Column* attr) const {
      return attributes.contains(attr);
   }
   ColumnSet& insert(const ColumnSet& other) {
      attributes.insert(other.attributes.begin(), other.attributes.end());
      return *this;
   }
   void remove(const ColumnSet& other) {
      for (const auto* elem : other.attributes) {
         attributes.erase(elem);
      }
   }
   bool intersects(const ColumnSet& others) const {
      for (const auto* x : attributes) {
         if (others.attributes.contains(x)) {
            return true;
         }
      }
      return false;
   }

   bool isSubsetOf(const ColumnSet& others) const {
      for (const auto* x : attributes) {
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
      auto& attributeManager = context->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
      for (const auto* x : attributes) {
         auto [scope, name] = attributeManager.getName(x);
         llvm::dbgs() << x << "(" << scope << "," << name << "),";
      }
   }
   ArrayAttr asRefArrayAttr(MLIRContext* context) {
      auto& attributeManager = context->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

      std::vector<Attribute> refAttrs;
      for (const auto* attr : attributes) {
         refAttrs.push_back(attributeManager.createRef(attr));
      }
      return ArrayAttr::get(context, refAttrs);
   }
   static ColumnSet fromArrayAttr(ArrayAttr arrayAttr) {
      ColumnSet res;
      for (const auto attr : arrayAttr) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>()) {
            res.insert(&attrRef.getColumn());
         } else if (auto attrDef = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>()) {
            res.insert(&attrDef.getColumn());
         }
      }
      return res;
   }
   static ColumnSet from(mlir::relalg::ColumnRefAttr attrRef) {
      ColumnSet res;
      res.insert(&attrRef.getColumn());
      return res;
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_COLUMNSET_H
