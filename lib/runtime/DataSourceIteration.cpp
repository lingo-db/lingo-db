#include "runtime/DataSourceIteration.h"
#include "json.h"
#include <iostream>
#include <iterator>

#include "runtime/Database.h"
#include "utility/Tracer.h"
#include <arrow/table.h>
#include <arrow/array.h>
#include <oneapi/tbb.h>
namespace {
static utility::Tracer::Event processMorsel("DataSourceIteration", "processMorsel");
static utility::Tracer::Event tbbForEach("DataSourceIteration", "tbbForEach");

static utility::Tracer::Event processMorselSingle("DataSourceIteration", "processMorselSingle");

static utility::Tracer::Event cleanupTLS("DataSourceIteration", "cleanup");
static utility::Tracer::Event tableScan("DataSourceIteration", "tableScan");

static void access(std::vector<size_t> colIds, runtime::RecordBatchInfo* info, const std::shared_ptr<arrow::RecordBatch>& currChunk) {
   for (size_t i = 0; i < colIds.size(); i++) {
      auto colId = colIds[i];
      runtime::ColumnInfo& colInfo = info->columnInfo[i];
      size_t off = currChunk->column_data(colId)->offset;
      colInfo.offset = off;
      colInfo.validMultiplier = currChunk->column_data(colId)->buffers[0] ? 1 : 0;
      colInfo.validBuffer = runtime::RecordBatchInfo::getBuffer(currChunk.get(), colId, 0);
      colInfo.dataBuffer = runtime::RecordBatchInfo::getBuffer(currChunk.get(), colId, 1);
      colInfo.varLenBuffer = runtime::RecordBatchInfo::getBuffer(currChunk.get(), colId, 2);
   }
   info->numRows = currChunk->num_rows();
}
class RecordBatchTableSource : public runtime::DataSource {
   const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches;
   std::unordered_map<std::string, size_t> memberToColumnId;

   public:
   RecordBatchTableSource(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, std::unordered_map<std::string, size_t> memberToColumnId) : batches(batches), memberToColumnId(memberToColumnId) {}
   void iterate(bool parallel, std::vector<size_t> colIds, const std::function<void(runtime::RecordBatchInfo*)>& cb) override {
      if (parallel) {
         tbb::enumerable_thread_specific<runtime::RecordBatchInfo*> batchInfo([&]() { return reinterpret_cast<runtime::RecordBatchInfo*>(malloc(sizeof(runtime::RecordBatchInfo) + sizeof(runtime::ColumnInfo) * colIds.size())); });
         utility::Tracer::Trace tbbTrace(tbbForEach);
         tbb::parallel_for_each(batches.begin(), batches.end(), [&](const std::shared_ptr<arrow::RecordBatch>& batch) {
            utility::Tracer::Trace trace(processMorsel);
            access(colIds, batchInfo.local(), batch);
            cb(batchInfo.local());
            trace.stop();
         });
         tbbTrace.stop();
         utility::Tracer::Trace cleanUpTrace(cleanupTLS);

         for (auto* bI : batchInfo) {
            if (bI) {
               free(bI);
            }
         }
         cleanUpTrace.stop();
      } else {
         auto* batchInfo = reinterpret_cast<runtime::RecordBatchInfo*>(malloc(sizeof(runtime::RecordBatchInfo) + sizeof(runtime::ColumnInfo) * colIds.size()));

         for (const auto& batch : batches) {
            utility::Tracer::Trace trace(processMorselSingle);
            access(colIds, batchInfo, batch);
            cb(batchInfo);
            trace.stop();
         }
         free(batchInfo);
      }
   }
   size_t getColumnId(std::string member) override {
      if (!memberToColumnId.contains(member)) {
         throw std::runtime_error("data source: invalid member");
      }
      return memberToColumnId[member];
   }
};
class ArrowTableSource : public runtime::DataSource {
   std::unordered_map<std::string, size_t> memberToColumnId;
   arrow::Table& table;

   public:
   ArrowTableSource(arrow::Table& table, std::unordered_map<std::string, size_t> memberToColumnId) : memberToColumnId(memberToColumnId), table(table) {}
   void iterate(bool parallel, std::vector<size_t> colIds, const std::function<void(runtime::RecordBatchInfo*)>& cb) override {
      auto* batchInfo = reinterpret_cast<runtime::RecordBatchInfo*>(malloc(sizeof(runtime::RecordBatchInfo) + sizeof(runtime::ColumnInfo) * colIds.size()));
      std::shared_ptr<arrow::RecordBatch> nextChunk;
      arrow::TableBatchReader reader(table);
      while (reader.ReadNext(&nextChunk) == arrow::Status::OK()) {
         if (nextChunk) {
            access(colIds, batchInfo, nextChunk);
            cb(batchInfo);
         } else {
            break;
         }
         nextChunk.reset();
      }
      free(batchInfo);
   }
   size_t getColumnId(std::string member) override {
      if (!memberToColumnId.contains(member)) {
         throw std::runtime_error("data source: invalid member");
      }
      return memberToColumnId[member];
   }
};
} // end namespace

void runtime::DataSourceIteration::end(DataSourceIteration* iteration) {
   delete iteration;
}
uint64_t getTableColumnId(const std::shared_ptr<arrow::Table>& table, std::string columnName) {
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
runtime::DataSourceIteration::DataSourceIteration(DataSource* dataSource, const std::vector<size_t>& colIds) : dataSource(dataSource), colIds(colIds) {
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
   if (executionContext->db->recordBatchesAvailable()) {
      return new RecordBatchTableSource(executionContext->db->getRecordBatches(tableName), memberToColumnId);
   }
   return new ArrowTableSource(*table, memberToColumnId);
}

void runtime::DataSourceIteration::iterate(bool parallel, void (*forEachChunk)(runtime::RecordBatchInfo*, void*), void* context) {
   utility::Tracer::Trace trace(tableScan);
   dataSource->iterate(parallel, colIds, [context, forEachChunk](runtime::RecordBatchInfo* recordBatchInfo) {
      forEachChunk(recordBatchInfo, context);
   });
   trace.stop();
}