#ifndef LINGODB_CATALOG_DEFS_H
#define LINGODB_CATALOG_DEFS_H
#include "TableCatalogEntry.h"
namespace lingodb::catalog {
struct CreateTableDef {
   std::string name;
   std::vector<Column> columns;
   std::vector<std::string> primaryKey;

   void serialize(utility::Serializer& serializer) const;
   static CreateTableDef deserialize(utility::Deserializer& deserializer);
};
} // namespace lingodb::catalog

#endif //LINGODB_CATALOG_DEFS_H
