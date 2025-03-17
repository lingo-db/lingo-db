#ifndef LINGODB_CATALOG_TABLECATALOGENTRY_H
#define LINGODB_CATALOG_TABLECATALOGENTRY_H
#include "Catalog.h"
#include "Types.h"
#include "lingodb/scheduler/Task.h"
#include <vector>

#include <arrow/type_fwd.h>
namespace lingodb::catalog {
class Sample {
   std::shared_ptr<arrow::RecordBatch> sampleData;

   public:
   Sample(std::shared_ptr<arrow::RecordBatch> sampleData) : sampleData(sampleData) {}
   std::shared_ptr<arrow::RecordBatch> getSampleData() const { return sampleData; }
   void serialize(utility::Serializer& serializer) const;
   static std::unique_ptr<Sample> deserialize(utility::Deserializer& deserializer);
};
class Column {
   std::string columnName;
   Type logicalType;
   bool isNullable;
   std::shared_ptr<arrow::DataType> arrowType;

   public:
   Column(std::string columnName, Type type, bool isNullable, std::shared_ptr<arrow::DataType> arrowType) : columnName(columnName), logicalType(logicalType), isNullable(isNullable), arrowType(arrowType) {}

   Type getLogicalType() const { return logicalType; }
   std::string getColumnName() const { return columnName; }
   bool getIsNullable() const { return isNullable; }
   std::shared_ptr<arrow::DataType> getArrowType() const { return arrowType; }
   void setArrowType(std::shared_ptr<arrow::DataType> type) { arrowType = type; }
};

class ColumnStatistics {
   std::optional<size_t> numDistinctValues;

   public:
   ColumnStatistics(std::optional<size_t> numDistinctValues) : numDistinctValues(numDistinctValues) {}
   std::optional<size_t> getNumDistinctValues() { return numDistinctValues; }
   void serialize(utility::Serializer& serializer) const;
   static std::unique_ptr<ColumnStatistics> deserialize(utility::Deserializer& deserializer);
};
class TableMetaDataProvider {
   public:
   virtual Sample& getSample() const = 0;
   virtual std::vector<std::string> getColumnNames() const = 0;
   virtual ColumnStatistics& getColumnStatistics(std::string column) const = 0;
   virtual size_t getNumRows() const = 0;
   virtual std::vector<std::string> getPrimaryKey() const = 0;
   void serialize(utility::Serializer& serializer) const;
   static std::unique_ptr<TableMetaDataProvider> deserialize(utility::Deserializer& deserializer);
};
class TableCatalogEntry : public CatalogEntry, TableMetaDataProvider {
   std::string name;
   std::vector<Column> columns;
   std::vector<std::string> primaryKey;

   public:
   TableCatalogEntry(std::string name, std::vector<Column> columns, std::vector<std::string> primaryKey) : CatalogEntry(CatalogEntryType::TABLE_ENTRY), name(name), columns(columns), primaryKey(primaryKey) {}
   std::string getName() override { return name; }
   void serializeEntry(lingodb::utility::Serializer& serializer) const override;
   static std::unique_ptr<TableCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
   std::vector<std::string> getColumnNames() const override {
      std::vector<std::string> columnNames;
      for (const auto& column : columns) {
         columnNames.push_back(column.getColumnName());
      }
      return columnNames;
   }
   virtual std::unique_ptr<scheduler::Task> createScanTask() = 0;
};

class InMemoryTableCatalogEntry : public TableCatalogEntry {};
class PersistedTableCatalogEntry : public TableCatalogEntry {};
} //namespace lingodb::catalog

#endif //LINGODB_CATALOG_TABLECATALOGENTRY_H
