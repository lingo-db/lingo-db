#ifndef LINGODB_COMPILER_FRONTEND_COLUMN_SEMANTIC_H
#define LINGODB_COMPILER_FRONTEND_COLUMN_SEMANTIC_H

#include "frontend_type.h"
#include "lingodb/catalog/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"

#include <string>
#include <vector>
namespace lingodb::ast {
class BoundFunctionExpression;

class ColumnReference {
   public:
   std::string scope;
   NullableType resultType;
   std::string name;
   std::string displayName{};
   /**
    * Indicates whether the column is selectable. Set to false when the column is not included
    * in the GROUP BY clause.
   */
   bool isSelectable = true;
   ColumnReference(std::string scope, NullableType resultType, std::string name) : scope(scope), resultType(resultType), name(name) {}
   ColumnReference(std::string scope, catalog::Column c) : scope(scope), resultType(NullableType(c.getLogicalType(), c.getIsNullable())), name(c.getColumnName()), displayName(c.getColumnName()) {}

   bool operator==(const ColumnReference& other) const {
      return other.name == name && other.scope == scope && other.displayName == displayName;
   }

   compiler::dialect::tuples::ColumnRefAttr createRef(mlir::OpBuilder& builder, compiler::dialect::tuples::ColumnManager& attrManager) {
      auto ref = attrManager.createRef(this->scope, name);
      ref.getColumn().type = resultType.toMlirType(builder.getContext());
      return ref;
   };

   compiler::dialect::tuples::ColumnDefAttr createDef(mlir::OpBuilder& builder, compiler::dialect::tuples::ColumnManager& attrManager) {
      auto def = attrManager.createDef(this->scope, name);
      def.getColumn().type = resultType.toMlirType(builder.getContext());
      return def;
   };
   compiler::dialect::tuples::ColumnDefAttr createDef(mlir::OpBuilder& builder, compiler::dialect::tuples::ColumnManager& attrManager, mlir::Attribute fromExisting) {
      auto def = attrManager.createDef(this->scope, name, fromExisting);
      def.getColumn().type = resultType.toMlirType(builder.getContext());
      return def;
   };
};

struct ColumnRefHash {
   std::size_t operator()(const std::pair<std::shared_ptr<ast::ColumnReference>, size_t> p) const noexcept {
      std::size_t h1 = std::hash<std::string>{}(p.first ? p.first->name : std::string{});
      std::size_t h2 = std::hash<decltype(ast::ColumnReference::scope)>{}(p.first ? p.first->scope : decltype(ast::ColumnReference::scope){});
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
   }
};

struct ColumnRefEq {
   using is_transparent = void;
   bool operator()(const std::pair<std::shared_ptr<ast::ColumnReference>, size_t> a,
                   const std::pair<std::shared_ptr<ast::ColumnReference>, size_t>
                      b) const noexcept {
      return a.first->scope == b.first->scope && a.first->name == b.first->name;
   }
};

struct TargetInfo {
   void add(std::shared_ptr<ColumnReference> entry) {
      targetColumns.push_back(std::move(entry));
   }

   std::shared_ptr<ColumnReference> getTargetColumn(size_t i) {
      return targetColumns[i];
   }
   std::shared_ptr<ColumnReference>& operator[](size_t i) {
      assert(targetColumns.size() > i);
      return targetColumns[i];
   }
   const std::vector<std::shared_ptr<ColumnReference>> getTargetColumns() const {
      std::vector<std::shared_ptr<ColumnReference>> result;
      for (auto w : targetColumns) {
         result.push_back(w);
      }

      return result;
   }

   void setTargetColumns(std::vector<std::shared_ptr<ColumnReference>> colums) {
      targetColumns = std::vector<std::shared_ptr<ColumnReference>>(colums.begin(), colums.end());
   }

   void clear() {
      targetColumns.clear();
   }
   size_t size() {
      return targetColumns.size();
   }
   void resize(size_t size) {
      targetColumns.resize(size);
   }

   void eraseif(std::function<bool(const std::shared_ptr<ColumnReference>& entry)> fn) {
      std::erase_if(targetColumns, [&](const std::weak_ptr<ColumnReference>& entry) {
         return fn(entry.lock());
      });
   }

   private:
   std::vector<std::shared_ptr<ColumnReference>> targetColumns;
};
} // namespace lingodb::ast
#endif
