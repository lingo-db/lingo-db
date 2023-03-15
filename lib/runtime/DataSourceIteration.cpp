#include "runtime/DataSourceIteration.h"
#include "json.h"
#include <arrow/array.h>
#include <arrow/table.h>

class ArrowTableSourceIterator : public runtime::DataSourceIterator {
   arrow::TableBatchReader reader;

   public:
   ArrowTableSourceIterator(arrow::Table& table) : reader(table) {}
   std::shared_ptr<arrow::RecordBatch> getNext() override {
      std::shared_ptr<arrow::RecordBatch> nextChunk;
      if (reader.ReadNext(&nextChunk) != arrow::Status::OK()) {
         nextChunk.reset();
      }
      return nextChunk;
   }
};
class ArrowTableSource : public runtime::DataSource {
   std::unordered_map<std::string, size_t> memberToColumnId;
   arrow::Table& table;

   public:
   ArrowTableSource(arrow::Table& table, std::unordered_map<std::string, size_t> memberToColumnId) : memberToColumnId(memberToColumnId), table(table) {}
   std::shared_ptr<runtime::DataSourceIterator> getIterator() override {
      return std::make_shared<ArrowTableSourceIterator>(table);
   }
   size_t getColumnId(std::string member) override {
      if (!memberToColumnId.contains(member)) {
         throw std::runtime_error("data source: invalid member");
      }
      return memberToColumnId[member];
   }
};

bool runtime::DataSourceIteration::isValid() {
   return !!currChunk;
}
void runtime::DataSourceIteration::next() {
   currChunk = iterator->getNext();
}
void runtime::DataSourceIteration::access(RecordBatchInfo* info) {
   for (size_t i = 0; i < colIds.size(); i++) {
      auto colId = colIds[i];
      ColumnInfo& colInfo = info->columnInfo[i];
      size_t off = currChunk->column_data(colId)->offset;
      colInfo.offset = off;
      colInfo.validMultiplier = currChunk->column_data(colId)->buffers[0] ? 1 : 0;
      colInfo.validBuffer = RecordBatchInfo::getBuffer(currChunk.get(), colId, 0);
      colInfo.dataBuffer = RecordBatchInfo::getBuffer(currChunk.get(), colId, 1);
      colInfo.varLenBuffer = RecordBatchInfo::getBuffer(currChunk.get(), colId, 2);
   }
   info->numRows = currChunk->num_rows();
}
void runtime::DataSourceIteration::end(DataSourceIteration* iteration) {
   delete iteration;
}
uint64_t getTableColumnId(std::shared_ptr<arrow::Table> table, std::string columnName) {
   auto columnNames = table->ColumnNames();
   size_t columnId = 0;
   for (auto column : columnNames) {
      if (column == columnName) {
         return columnId;
      }
      columnId++;
   }
   throw std::runtime_error("column not found: " + columnName);
}
runtime::DataSourceIteration* runtime::DataSourceIteration::init(DataSource* dataSource, runtime::VarLen32 members) {
   nlohmann::json descr = nlohmann::json::parse(members.str());
   std::vector<size_t> colIds;
   for (std::string c : descr.get<nlohmann::json::array_t>()) {
      colIds.push_back(dataSource->getColumnId(c));
   }
   return new DataSourceIteration(dataSource, colIds);
}
runtime::DataSourceIteration::DataSourceIteration(DataSource* dataSource, const std::vector<size_t>& colIds) : dataSource(dataSource), iterator(dataSource->getIterator()), colIds(colIds) {
   currChunk = iterator->getNext();
}

runtime::DataSource* runtime::DataSource::get(runtime::ExecutionContext* executionContext, runtime::VarLen32 description) {
   nlohmann::json descr = nlohmann::json::parse(description.str());
   std::string tableName = descr["table"];
   if (!executionContext->db) {
      throw std::runtime_error("no database attached");
   }
   auto table = (executionContext)->db->getTable(tableName);
   if (!table) {
      throw std::runtime_error("could not find table");
   }
   std::unordered_map<std::string, size_t> memberToColumnId;
   for (auto m : descr["mapping"].get<nlohmann::json::object_t>()) {
      memberToColumnId[m.first] = getTableColumnId(table, m.second.get<std::string>());
   }
   return new ArrowTableSource(*table.get(), memberToColumnId);
}
