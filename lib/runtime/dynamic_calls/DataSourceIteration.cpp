#include "runtime/DataSourceIteration.h"
#include "json.h"
#include <arrow/array.h>
#include <arrow/table.h>

class ArrowTableSource : public runtime::DataSource {
   arrow::TableBatchReader reader;

   public:
   ArrowTableSource(arrow::Table& table) : reader(table) {}
   std::shared_ptr<arrow::RecordBatch> getNext() override {
      std::shared_ptr<arrow::RecordBatch> nextChunk;
      if (reader.ReadNext(&nextChunk) != arrow::Status::OK()) {
         nextChunk.reset();
      }
      return nextChunk;
   }
};
bool runtime::DataSourceIteration::isValid() {
   return !!currChunk;
}
void runtime::DataSourceIteration::next() {
   currChunk = dataSource->getNext();
}
uint8_t* get_buffer(arrow::RecordBatch* batch, size_t columnId, size_t bufferId) {
   static uint8_t alternative = 0b11111111;
   if (batch->column_data(columnId)->buffers.size() > bufferId && batch->column_data(columnId)->buffers[bufferId]) {
      auto* buffer = batch->column_data(columnId)->buffers[bufferId].get();
      return (uint8_t*) buffer->address();
   } else {
      return &alternative; //always return valid pointer to at least one byte filled with ones
   }
}
void runtime::DataSourceIteration::access(RecordBatchInfo* info) {
   for (size_t i = 0; i < colIds.size(); i++) {
      auto colId = colIds[i];
      ColumnInfo& colInfo = info->columnInfo[i];
      size_t off = currChunk->column_data(colId)->offset;
      colInfo.offset = off;
      colInfo.validMultiplier = currChunk->column_data(colId)->buffers[0] ? 1 : 0;
      colInfo.validBuffer = get_buffer(currChunk.get(), colId, 0);
      colInfo.dataBuffer = get_buffer(currChunk.get(), colId, 1);
      colInfo.varLenBuffer = get_buffer(currChunk.get(), colId, 2);
   }
   info->numRows = currChunk->num_rows();
}
void runtime::DataSourceIteration::end(DataSourceIteration* iteration) {
   delete iteration;
}
uint64_t get_column_id(std::shared_ptr<arrow::Table> table, std::string columnName) {
   auto column_names = table->ColumnNames();
   size_t column_id = 0;
   for (auto column : column_names) {
      if (column == columnName) {
         return column_id;
      }
      column_id++;
   }
   throw std::runtime_error("column not found: " + columnName);
}
runtime::DataSourceIteration* runtime::DataSourceIteration::start(ExecutionContext* executionContext, runtime::VarLen32 description) {
   nlohmann::json descr = nlohmann::json::parse(description.str());
   std::string tableName = descr["table"];
   if (!executionContext->db) {
      throw std::runtime_error("no database attached");
   }
   auto table = (executionContext)->db->getTable(tableName);
   if (!table) {
      throw std::runtime_error("could not find table");
   }
   std::vector<size_t> colIds;
   for (std::string c : descr["columns"]) {
      colIds.push_back(get_column_id(table, c));
   }
   return new DataSourceIteration(std::make_shared<ArrowTableSource>(*table.get()), colIds);
}
runtime::DataSourceIteration::DataSourceIteration(const std::shared_ptr<DataSource>& dataSource, const std::vector<size_t>& colIds) : dataSource(dataSource), colIds(colIds) {
   currChunk = dataSource->getNext();
}
