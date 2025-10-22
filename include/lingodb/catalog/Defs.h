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

struct CreateFunctionDef {
   std::string name;
   std::string language;
   std::string code;
   Type returnType;
   std::vector<Type> argumentTypes;
   CreateFunctionDef(std::string name, std::string language, std::string code, Type returnType, std::vector<Type> argumentTypes) : name(std::move(name)), language(std::move(language)), code(std::move(code)), returnType(std::move(returnType)), argumentTypes(std::move(argumentTypes)) {};
   void serialize(utility::Serializer& serializer) const;
   static CreateFunctionDef deserialize(utility::Deserializer& deserializer);
};
} // namespace lingodb::catalog

#endif //LINGODB_CATALOG_DEFS_H
