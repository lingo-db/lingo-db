#ifndef LINGODB_CATALOG_TABLECATALOGENTRY_H
#define LINGODB_CATALOG_TABLECATALOGENTRY_H
#include "Catalog.h"
#include "Types.h"
#include <vector>
namespace lingodb::catalog {
class TableColumn {
   std::string columnName;
   Type type;
   bool isNullable;

   public:
   TableColumn(std::string columnName, Type type, bool isNullable = true) : columnName(columnName), type(type), isNullable(isNullable) {}
   void serialize(lingodb::utility::Serializer& serializer) const;
   static TableColumn deserialize(lingodb::utility::Deserializer& deserializer);
};
class ColumnStatistics{
   double estimateDistinctValues();
};
class TableStatistics{
   size_t getNumRows();
   void getSamples();
};

class TableCatalogEntry : public CatalogEntry {
   std::string name;
   std::vector<TableColumn> columns;
   std::vector<std::string> primaryKey;

   public:
   TableCatalogEntry(std::string name, std::vector<TableColumn> columns, std::vector<std::string> primaryKey) : CatalogEntry(CatalogEntryType::TABLE_ENTRY), name(name), columns(columns), primaryKey(primaryKey) {}
   void serializeEntry(lingodb::utility::Serializer& serializer) const override;
   static std::unique_ptr<TableCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
   std::string getName() override { return name; }
};
} //namespace lingodb::catalog

#endif //LINGODB_CATALOG_TABLECATALOGENTRY_H
