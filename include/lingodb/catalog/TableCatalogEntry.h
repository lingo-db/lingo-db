#ifndef LINGODB_CATALOG_TABLECATALOGENTRY_H
#define LINGODB_CATALOG_TABLECATALOGENTRY_H
#include "Catalog.h"
#include "Column.h"
#include "MetaData.h"
#include "Types.h"

#include <vector>

#include <arrow/type_fwd.h>
namespace lingodb::runtime {
class LingoDBTable;
class TableStorage;
} // namespace lingodb::runtime
namespace lingodb::catalog {
struct CreateTableDef;

class TableCatalogEntry : public CatalogEntry, public TableMetaDataProvider {
   protected:
   std::string name;
   std::vector<Column> columns;
   std::vector<std::string> primaryKey;
   //todo: integrate properly
   std::vector<std::string> indices;

   public:
   static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::LINGODB_TABLE_ENTRY};
   TableCatalogEntry(CatalogEntryType entryType, std::string name, std::vector<Column> columns, std::vector<std::string> primaryKey, std::vector<std::string> indices) : CatalogEntry(entryType), name(name), columns(columns), primaryKey(primaryKey), indices(indices) {}
   std::string getName() override { return name; }
   std::vector<std::string> getColumnNames() const override {
      std::vector<std::string> columnNames;
      for (const auto& column : columns) {
         columnNames.emplace_back(column.getColumnName());
      }
      return columnNames;
   }
   std::vector<Column> getColumns() const { return columns; }
   std::vector<std::string> getPrimaryKey() const override { return primaryKey; }
   std::vector<std::pair<std::string, std::vector<std::string>>> getIndices() const override;
   void addIndex(std::string indexName) { indices.emplace_back(std::move(indexName)); }
   virtual runtime::TableStorage& getTableStorage() = 0;
};

class LingoDBTableCatalogEntry : public TableCatalogEntry {
   std::unique_ptr<runtime::LingoDBTable> impl;

   public:
   LingoDBTableCatalogEntry(std::string name, std::vector<Column> columns, std::vector<std::string> primaryKey, std::vector<std::string> indices, std::unique_ptr<runtime::LingoDBTable> impl);

   static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::LINGODB_TABLE_ENTRY};
   void serializeEntry(lingodb::utility::Serializer& serializer) const override;
   static std::shared_ptr<LingoDBTableCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
   const Sample& getSample() const override;
   const ColumnStatistics& getColumnStatistics(std::string_view column) const override;
   size_t getNumRows() const override;
   ~LingoDBTableCatalogEntry() override = default;
   runtime::TableStorage& getTableStorage() override;
   virtual void flush() override;
   virtual void ensureFullyLoaded() override;
   virtual void setShouldPersist(bool shouldPersist) override;
   virtual void setDBDir(std::string dbDir) override;
   static std::shared_ptr<LingoDBTableCatalogEntry> createFromCreateTable(const CreateTableDef& def);
};
} //namespace lingodb::catalog

#endif //LINGODB_CATALOG_TABLECATALOGENTRY_H
