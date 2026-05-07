#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_CREATE_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_CREATE_NODE_H

#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/frontend/ast/create_node.h"
#include "lingodb/compiler/frontend/frontend_type.h"

namespace lingodb::ast {
/**
 * Represents a column element inside a create table element
 * CREATE TABLE x (y int primary, z int). Both y and z are each represented by a BoundColumnElement
 */
class BoundColumnElement : public TableElement {
   public:
   BoundColumnElement(std::string name, NullableType datatype, bool primary)
      : TableElement(TableElementType::COLUMN), name(std::move(name)), datatype(datatype), primary(primary) {}

   std::string name;
   NullableType datatype;

   bool unique;
   bool primary;
};

// Bound form of `CREATE FUNCTION ... RETURNS <scalar>`.
class BoundCreateFunctionInfo : public CreateInfo {
   public:
   BoundCreateFunctionInfo(std::string functionName, bool replace, NullableType returnType)
      : CreateInfo(catalog::CatalogEntry::CatalogEntryType::C_FUNCTION_ENTRY, std::move(""), std::move(""), false), functionName(functionName), replace(replace), returnType(returnType) {}

   std::string functionName;
   bool replace;
   std::string aliasOf;

   std::string language;
   std::string code;
   NullableType returnType;
   std::vector<std::pair<std::string, catalog::Type>> argumentTypes;
   std::vector<std::string> argumentNames;

   std::vector<std::pair<std::string, std::string>> options;
};

// Bound form of `CREATE FUNCTION ... RETURNS TABLE(...)`.
class BoundCreateTableFunctionInfo : public CreateInfo {
   public:
   BoundCreateTableFunctionInfo(std::string functionName, bool replace)
      : CreateInfo(catalog::CatalogEntry::CatalogEntryType::TABLE_FUNCTION_ENTRY, std::move(""), std::move(""), false), functionName(functionName), replace(replace) {}

   std::string functionName;
   bool replace;

   std::string language;
   std::string code;
   //! Declared input table: parameter name + per-column (name, type) schema.
   std::string inputTableName;
   std::vector<std::pair<std::string, catalog::Type>> inputColumns;
   //! Scalar arguments after the input table.
   std::vector<std::pair<std::string, catalog::Type>> argumentTypes;
   //! Per-column output schema declared in `RETURNS TABLE(...)`.
   std::vector<std::pair<std::string, catalog::Type>> returnColumns;

   std::vector<std::pair<std::string, std::string>> options;
};

} // namespace lingodb::ast
#endif
