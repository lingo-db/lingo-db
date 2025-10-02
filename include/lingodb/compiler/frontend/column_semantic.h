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
   std::vector<std::shared_ptr<ColumnReference>> targetColumns;
   void add(std::shared_ptr<ColumnReference> entry) {
      targetColumns.push_back(std::move(entry));
   }
};
} // namespace lingodb::ast
#endif
