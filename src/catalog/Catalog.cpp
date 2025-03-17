#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"
#include <arrow/buffer.h>
#include <arrow/type.h>

#include <arrow/ipc/api.h>

#include <arrow/io/api.h>

namespace lingodb::catalog {
std::unique_ptr<Catalog> Catalog::deserialize(lingodb::utility::Deserializer& deSerializer) {
   auto res = std::make_unique<Catalog>();
   res->entries = deSerializer.readProperty<std::unordered_map<std::string, std::unique_ptr<CatalogEntry>>>(1);
   return res;
}
void Catalog::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entries);
}

void CatalogEntry::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entryType);
   serializeEntry(serializer);
}
std::unique_ptr<CatalogEntry> CatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto entryType = deserializer.readProperty<CatalogEntryType>(1);
   switch (entryType) {
      case CatalogEntryType::INVALID_ENTRY:
         return nullptr;
      case CatalogEntryType::TABLE_ENTRY:
         return TableCatalogEntry::deserialize(deserializer);
   }
}
void TableCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(2, name);
   serializer.writeProperty(3, columns.size());
   arrow::FieldVector fields;
   for (const auto& column : columns) {
      serializer.writeProperty(4, column.getColumnName());
      serializer.writeProperty(5, column.getLogicalType());
      serializer.writeProperty(6, column.getIsNullable());
      fields.push_back(arrow::field(column.getColumnName(), column.getArrowType()));
   }
   arrow::Schema s(fields);
   auto res = arrow::ipc::SerializeSchema(s).ValueOrDie();
   serializer.writeProperty(7, std::string_view((char*) res->data(), res->size()));
   serializer.writeProperty(8, primaryKey);
}
std::unique_ptr<TableCatalogEntry> TableCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto numColumns = deserializer.readProperty<size_t>(3);
   std::vector<Column> columns;
   for (size_t i = 0; i < numColumns; i++) {
      auto columnName = deserializer.readProperty<std::string>(4);
      auto logicalType = deserializer.readProperty<Type>(5);
      auto isNullable = deserializer.readProperty<bool>(6);
      columns.push_back(Column(columnName, logicalType, isNullable, {}));
   }
   auto schemaData = deserializer.readProperty<std::string>(7);
   arrow::ipc::DictionaryMemo dictMemo;
   auto bufferReader = arrow::io::BufferReader::FromString(schemaData);
   auto schema = arrow::ipc::ReadSchema(bufferReader.get(), &dictMemo).ValueOrDie();
   for (size_t i = 0; i < columns.size(); i++) {
      columns[i].setArrowType(schema->field(i)->type());
   }
   auto primaryKey = deserializer.readProperty<std::vector<std::string>>(8);
   return std::make_unique<TableCatalogEntry>(name, columns, primaryKey);
}

void Sample::serialize(utility::Serializer& serializer) const {
   std::shared_ptr<arrow::ResizableBuffer> buffer = arrow::AllocateResizableBuffer(0).ValueOrDie();
   std::unique_ptr<arrow::io::BufferOutputStream> bufferOutputStream = std::make_unique<arrow::io::BufferOutputStream>(buffer);
   std::shared_ptr<arrow::ipc::RecordBatchWriter> recordBatchWriter = arrow::ipc::MakeStreamWriter(bufferOutputStream.get(), sampleData->schema()).ValueOrDie();

   if (!recordBatchWriter->WriteRecordBatch(*sampleData).ok() || !recordBatchWriter->Close().ok()) {
      throw std::runtime_error("MetaData: Failed to write record batch");
   }
   auto resBuffer = bufferOutputStream->Finish().ValueOrDie();
   serializer.writeProperty(1, std::string_view(*resBuffer));
}
std::unique_ptr<Sample> Sample::deserialize(utility::Deserializer& deserializer) {
   auto data = deserializer.readProperty<std::string>(1);
   auto bufferReader = arrow::io::BufferReader::FromString(data);
   auto recordBatchReader = arrow::ipc::RecordBatchStreamReader::Open(bufferReader.get()).ValueOrDie();
   std::shared_ptr<arrow::RecordBatch> batch;
   if (!recordBatchReader->ReadNext(&batch).ok()) {
      throw std::runtime_error("Failed to deserialize sample");
   }
   return std::make_unique<Sample>(batch);
}
std::unique_ptr<ColumnStatistics> ColumnStatistics::deserialize(utility::Deserializer& deserializer) {
   auto numDistinctValues = deserializer.readProperty<std::optional<size_t>>(1);
   return std::make_unique<ColumnStatistics>(numDistinctValues);
}
void ColumnStatistics::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, numDistinctValues);
}

void TableMetaDataProvider::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, getSample());
   serializer.writeProperty(2, getNumRows());
   serializer.writeProperty(3, getPrimaryKey());
   serializer.writeProperty(4, getColumnNames().size());
   for (const auto& column : getColumnNames()) {
      serializer.writeProperty(5, column);
      serializer.writeProperty(6, getColumnStatistics(column));
   }
}

class StoredTableMetaData : public TableMetaDataProvider {
   std::unique_ptr<Sample> sample;
   std::vector<std::string> primaryKey;
   std::vector<std::string> columnNames;
   std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics;
   size_t numRows;

   public:
   StoredTableMetaData(std::unique_ptr<Sample> sample, size_t numRows, std::vector<std::string> primaryKey, std::vector<std::string> columnNames, std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics) : sample(std::move(sample)), numRows(numRows), primaryKey(std::move(primaryKey)), columnNames(std::move(columnNames)), columnStatistics(std::move(columnStatistics)) {}
   Sample& getSample() const override { return *sample; }
   size_t getNumRows() const override { return numRows; }
   std::vector<std::string> getPrimaryKey() const override { return primaryKey; }
   std::vector<std::string> getColumnNames() const override { return columnNames; }
   ColumnStatistics& getColumnStatistics(std::string column) const override {
      for (size_t i = 0; i < columnNames.size(); i++) {
         if (columnNames[i] == column) {
            return *columnStatistics[i];
         }
      }
      throw std::runtime_error("MetaData: Column not found");
   }
};

std::unique_ptr<TableMetaDataProvider> TableMetaDataProvider::deserialize(utility::Deserializer& deserializer) {
   auto sample = deserializer.readProperty<std::unique_ptr<Sample>>(1);
   auto numRows = deserializer.readProperty<size_t>(2);
   auto primaryKey = deserializer.readProperty<std::vector<std::string>>(3);
   auto numColumns = deserializer.readProperty<size_t>(4);
   std::vector<std::string> columnNames;
   std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics;
   for (size_t i = 0; i < numColumns; i++) {
      columnNames.push_back(deserializer.readProperty<std::string>(5));
      columnStatistics.push_back(deserializer.readProperty<std::unique_ptr<ColumnStatistics>>(6));
   }
   return std::make_unique<StoredTableMetaData>(std::move(sample), numRows, std::move(primaryKey), std::move(columnNames), std::move(columnStatistics));
}

} // namespace lingodb::catalog