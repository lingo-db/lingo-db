#pragma once
#include "lingodb/catalog/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"

#include <string>
#include <vector>
namespace lingodb::ast {
class BoundFunctionExpression;

class NamedResult {
   public:
   std::string scope;
   catalog::NullableType resultType;
   std::string name;
   //TODO find better name
   std::string displayName{};
   NamedResult(std::string scope, catalog::NullableType resultType, std::string name) : scope(scope), resultType(resultType), name(name) {}
   NamedResult(std::string scope, catalog::Column c) : scope(scope), resultType(catalog::NullableType(c.getLogicalType(), c.getIsNullable())), name(c.getColumnName()), displayName(c.getColumnName()) {}

   bool operator==(const NamedResult& other) const {
      return other.name == name && other.scope == scope && other.displayName == displayName;
   }



   virtual compiler::dialect::tuples::ColumnRefAttr createRef(mlir::OpBuilder& builder, compiler::dialect::tuples::ColumnManager& attrManager) {
      auto ref = attrManager.createRef(this->scope, name);
      ref.getColumn().type = resultType.toMlirType(builder.getContext());
      return ref;
   };

   virtual compiler::dialect::tuples::ColumnDefAttr createDef(mlir::OpBuilder& builder, compiler::dialect::tuples::ColumnManager& attrManager) {
      auto def = attrManager.createDef(this->scope, name);
      def.getColumn().type = resultType.toMlirType(builder.getContext());
      return def;
   };
   virtual compiler::dialect::tuples::ColumnDefAttr createDef(mlir::OpBuilder& builder, compiler::dialect::tuples::ColumnManager& attrManager, mlir::Attribute fromExisting) {
      auto def = attrManager.createDef(this->scope, name, fromExisting);
      def.getColumn().type = resultType.toMlirType(builder.getContext());
      return def;
   };
};

struct TargetInfo {
   std::vector<std::shared_ptr<NamedResult>> targetColumns;
   void add(std::shared_ptr<NamedResult> entry) {
      targetColumns.push_back(std::move(entry));
   }
};
} // namespace lingodb::ast