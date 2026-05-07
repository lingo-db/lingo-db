#ifndef LINGODB_CATALOG_DEFS_H
#define LINGODB_CATALOG_DEFS_H
#include "Column.h"

#include <vector>

namespace lingodb::catalog {
struct CreateTableDef {
   std::string name;
   std::vector<Column> columns;
   std::vector<std::string> primaryKey;

   void serialize(utility::Serializer& serializer) const;
   static CreateTableDef deserialize(utility::Deserializer& deserializer);
};

// Wire-format payload for `CREATE FUNCTION ... RETURNS <scalar>`.
// Translator → RelationHelper::createScalarFunction → catalog.
struct CreateScalarFunctionDef {
   std::string name;
   std::string language;
   std::string code;
   Type returnType;
   std::vector<Type> argumentTypes;
   CreateScalarFunctionDef(std::string name, std::string language, std::string code,
                           Type returnType, std::vector<Type> argumentTypes)
      : name(std::move(name)), language(std::move(language)), code(std::move(code)),
        returnType(std::move(returnType)), argumentTypes(std::move(argumentTypes)) {}
   void serialize(utility::Serializer& serializer) const;
   static CreateScalarFunctionDef deserialize(utility::Deserializer& deserializer);
};

// Wire-format payload for `CREATE FUNCTION ... RETURNS TABLE(...)`.
struct CreateTableFunctionDef {
   std::string name;
   std::string language;
   std::string code;
   std::vector<Type> argumentTypes;
   std::vector<std::pair<std::string, Type>> returnColumns;
   CreateTableFunctionDef(std::string name, std::string language, std::string code,
                          std::vector<Type> argumentTypes,
                          std::vector<std::pair<std::string, Type>> returnColumns)
      : name(std::move(name)), language(std::move(language)), code(std::move(code)),
        argumentTypes(std::move(argumentTypes)), returnColumns(std::move(returnColumns)) {}
   void serialize(utility::Serializer& serializer) const;
   static CreateTableFunctionDef deserialize(utility::Deserializer& deserializer);
};
} // namespace lingodb::catalog

#endif //LINGODB_CATALOG_DEFS_H
