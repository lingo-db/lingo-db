
#ifndef LINGODB_COMPILER_DIALECT_RELALG_COLUMNSET_H
#define LINGODB_COMPILER_DIALECT_RELALG_COLUMNSET_H
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <lingodb/compiler/Dialect/TupleStream/Column.h>
#include <lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h>
namespace lingodb::compiler::dialect::relalg {
class ColumnSet {
   using attribute_set = llvm::SmallPtrSet<const lingodb::compiler::dialect::tuples::Column*, 8>;
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
   void insert(const lingodb::compiler::dialect::tuples::Column* attr) {
      attributes.insert(attr);
   }
   bool contains(const lingodb::compiler::dialect::tuples::Column* attr) const {
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
   void dump(mlir::MLIRContext* context) {
      auto& attributeManager = context->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()->getColumnManager();
      for (const auto* x : attributes) {
         auto [scope, name] = attributeManager.getName(x);
         llvm::dbgs() << x << "(" << scope << "," << name << "),";
      }
   }
   mlir::ArrayAttr asRefArrayAttr(mlir::MLIRContext* context) {
      auto& attributeManager = context->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<mlir::Attribute> refAttrs;
      for (const auto* attr : attributes) {
         refAttrs.push_back(attributeManager.createRef(attr));
      }
      return mlir::ArrayAttr::get(context, refAttrs);
   }
   static ColumnSet fromArrayAttr(mlir::ArrayAttr arrayAttr) {
      ColumnSet res;
      for (const auto attr : arrayAttr) {
         if (auto attrRef = mlir::dyn_cast_or_null<lingodb::compiler::dialect::tuples::ColumnRefAttr>(attr)) {
            res.insert(&attrRef.getColumn());
         } else if (auto attrDef = mlir::dyn_cast_or_null<lingodb::compiler::dialect::tuples::ColumnDefAttr>(attr)) {
            res.insert(&attrDef.getColumn());
         }
      }
      return res;
   }
   static ColumnSet from(dialect::tuples::ColumnRefAttr attrRef) {
      ColumnSet res;
      res.insert(&attrRef.getColumn());
      return res;
   }
   static ColumnSet from(const lingodb::compiler::dialect::tuples::Column* col) {
      ColumnSet res;
      res.insert(col);
      return res;
   }
};
} // namespace lingodb::compiler::dialect::relalg
#endif //LINGODB_COMPILER_DIALECT_RELALG_COLUMNSET_H
