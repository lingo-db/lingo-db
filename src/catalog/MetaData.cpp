#include "lingodb/catalog/MetaData.h"

#include "lingodb/utility/Serialization.h"

#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/type.h>

namespace lingodb::catalog {

ColumnStatistics ColumnStatistics::deserialize(utility::Deserializer& deserializer) {
   auto numDistinctValues = deserializer.readProperty<std::optional<size_t>>(1);
   return {numDistinctValues};
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
   serializer.writeProperty(7, getIndices());
}

class StoredTableMetaData : public TableMetaDataProvider {
   Sample sample;
   std::vector<std::string> primaryKey;
   std::vector<std::string> columnNames;
   std::vector<ColumnStatistics> columnStatistics;
   size_t numRows;
   std::vector<std::pair<std::string, std::vector<std::string>>> indices;

   public:
   StoredTableMetaData(const Sample& sample, size_t numRows, std::vector<std::string> primaryKey, std::vector<std::string> columnNames, std::vector<ColumnStatistics> columnStatistics, std::vector<std::pair<std::string, std::vector<std::string>>> indices) : sample(std::move(sample)), primaryKey(std::move(primaryKey)), columnNames(std::move(columnNames)), columnStatistics(std::move(columnStatistics)), numRows(numRows), indices(indices) {}
   const Sample& getSample() const override { return sample; }
   size_t getNumRows() const override { return numRows; }
   std::vector<std::string> getPrimaryKey() const override { return primaryKey; }
   std::vector<std::string> getColumnNames() const override { return columnNames; }
   const ColumnStatistics& getColumnStatistics(std::string column) const override {
      for (size_t i = 0; i < columnNames.size(); i++) {
         if (columnNames[i] == column) {
            return columnStatistics[i];
         }
      }
      throw std::runtime_error("MetaData: Column not found");
   }
   std::vector<std::pair<std::string, std::vector<std::string>>> getIndices() const override {
      return indices;
   }
   ~StoredTableMetaData() override = default;
};

std::shared_ptr<TableMetaDataProvider> TableMetaDataProvider::deserialize(utility::Deserializer& deserializer) {
   auto sample = deserializer.readProperty<Sample>(1);
   auto numRows = deserializer.readProperty<size_t>(2);
   auto primaryKey = deserializer.readProperty<std::vector<std::string>>(3);
   auto numColumns = deserializer.readProperty<size_t>(4);
   std::vector<std::string> columnNames;
   std::vector<ColumnStatistics> columnStatistics;
   for (size_t i = 0; i < numColumns; i++) {
      columnNames.push_back(deserializer.readProperty<std::string>(5));
      columnStatistics.push_back(deserializer.readProperty<ColumnStatistics>(6));
   }
   auto indices = deserializer.readProperty<std::vector<std::pair<std::string, std::vector<std::string>>>>(7);
   return std::make_shared<StoredTableMetaData>(sample, numRows, std::move(primaryKey), std::move(columnNames), columnStatistics, std::move(indices));
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
Sample Sample::deserialize(utility::Deserializer& deserializer) {
   auto data = deserializer.readProperty<std::string>(1);
   auto bufferReader = arrow::io::BufferReader::FromString(data);
   auto recordBatchReader = arrow::ipc::RecordBatchStreamReader::Open(bufferReader.get()).ValueOrDie();
   std::shared_ptr<arrow::RecordBatch> batch;
   if (!recordBatchReader->ReadNext(&batch).ok()) {
      throw std::runtime_error("Failed to deserialize sample");
   }
   return {batch};
}
Sample::Sample(std::shared_ptr<arrow::Schema> schema) {
   sampleData = arrow::RecordBatch::MakeEmpty(schema).ValueOrDie();
}
} // namespace lingodb::catalog