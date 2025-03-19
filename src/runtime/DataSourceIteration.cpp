#include "lingodb/runtime/DataSourceIteration.h"
#include "json.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"
#include <iterator>

#include "lingodb/utility/Tracer.h"

#include <arrow/array.h>
#include <arrow/table.h>
namespace utility = lingodb::utility;
namespace {

static utility::Tracer::Event tableScan("DataSourceIteration", "tableScan");

class TableSource : public lingodb::runtime::DataSource {
   lingodb::runtime::TableStorage& tableStorage;
   std::unordered_map<std::string, std::string> memberToColumn;

   public:
   TableSource(lingodb::runtime::TableStorage& tableStorage, std::unordered_map<std::string, std::string> memberToColumn) : tableStorage(tableStorage), memberToColumn(memberToColumn) {}
   void iterate(bool parallel, std::vector<std::string> members, const std::function<void(lingodb::runtime::RecordBatchInfo*)>& cb) override {
      std::vector<std::string> columns;
      for (const auto& member : members) {
         columns.push_back(memberToColumn.at(member));
      }
      auto scanTask = tableStorage.createScanTask({parallel, columns, cb});
      lingodb::scheduler::awaitChildTask(std::move(scanTask));
   }
};
/*
class RecordBatchSource : public lingodb::runtime::DataSource {
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches; //todo: efficiency?
   std::unordered_map<std::string, size_t> memberToColumnId;

   public:
   RecordBatchTableSource(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, std::unordered_map<std::string, size_t> memberToColumnId) : batches(batches), memberToColumnId(memberToColumnId) {}
   void iterate(bool parallel, std::vector<std::string> columns, const std::function<void(lingodb::runtime::RecordBatchInfo*)>& cb) override {
      if (parallel) {
         lingodb::scheduler::awaitChildTask(std::make_unique<ScanBatchesTask>(batches, colIds, cb));
      } else {
         auto* batchInfo = reinterpret_cast<lingodb::runtime::RecordBatchInfo*>(malloc(sizeof(lingodb::runtime::RecordBatchInfo) + sizeof(lingodb::runtime::ColumnInfo) * colIds.size()));

         for (const auto& batch : batches) {
            utility::Tracer::Trace trace(processMorselSingle);
            access(colIds, batchInfo, batch);
            cb(batchInfo);
            trace.stop();
         }
         free(batchInfo);
      }
   }
};
*/
std::vector<std::shared_ptr<arrow::RecordBatch>> toRecordBatches(std::shared_ptr<arrow::Table> table) {
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   arrow::TableBatchReader reader(table);
   reader.set_chunksize(20000);
   std::shared_ptr<arrow::RecordBatch> nextChunk;
   while (reader.ReadNext(&nextChunk) == arrow::Status::OK()) {
      if (nextChunk) {
         batches.push_back(nextChunk);
      } else {
         break;
      }
      nextChunk.reset();
   }
   return batches;
}
uint64_t getTableColumnId(const std::shared_ptr<arrow::Schema>& schema, std::string columnName) {
   auto columnNames = schema->field_names();
   size_t columnId = 0;
   for (auto column : columnNames) {
      if (column == columnName) {
         return columnId;
      }
      columnId++;
   }
   throw std::runtime_error("column not found: " + columnName);
}
} // end namespace

void lingodb::runtime::DataSourceIteration::end(DataSourceIteration* iteration) {
   delete iteration;
}

lingodb::runtime::DataSourceIteration* lingodb::runtime::DataSourceIteration::init(DataSource* dataSource, lingodb::runtime::VarLen32 rawMembers) {
   nlohmann::json descr = nlohmann::json::parse(rawMembers.str());
   std::vector<std::string> members;
   for (std::string c : descr.get<nlohmann::json::array_t>()) {
      members.push_back(c);
   }
   return new DataSourceIteration(dataSource, members);
}
lingodb::runtime::DataSourceIteration::DataSourceIteration(DataSource* dataSource, const std::vector<std::string>& members) : dataSource(dataSource), members(members) {
}

lingodb::runtime::DataSource* lingodb::runtime::DataSource::get(lingodb::runtime::VarLen32 description) {
   lingodb::runtime::ExecutionContext* executionContext = lingodb::runtime::getCurrentExecutionContext();
   nlohmann::json descr = nlohmann::json::parse(description.str());
   std::string tableName = descr["table"];
   auto& session = executionContext->getSession();
   if (auto maybeRelation = session.getCatalog()->getTypedEntry<catalog::TableCatalogEntry>(tableName)) {
      auto relation = maybeRelation.value();
      std::unordered_map<std::string, std::string> memberToColumn;
      for (auto m : descr["mapping"].get<nlohmann::json::object_t>()) {
         memberToColumn[m.first] = m.second.get<std::string>();
      }
      return new TableSource(relation->getTableStorage(), memberToColumn);
   } else {
      throw std::runtime_error("could not find relation");
   }
}
lingodb::runtime::DataSource* lingodb::runtime::DataSource::getFromTable(ArrowTable* arrowTable, lingodb::runtime::VarLen32 memberArray, lingodb::runtime::VarLen32 columnArray) {
   /*auto schema = arrowTable->get()->schema();
   std::unordered_map<std::string, size_t> memberToColumnId;
   nlohmann::json members = nlohmann::json::parse(memberArray.str());
   nlohmann::json columns = nlohmann::json::parse(columnArray.str());

   for (size_t i = 0; i < members.size(); i++) {
      memberToColumnId[members[i]] = arrowTable->get()->schema()->GetFieldIndex(columns[i]);
   }
   return new RecordBatchTableSource(toRecordBatches(arrowTable->get()), memberToColumnId);*/
}

void lingodb::runtime::DataSourceIteration::iterate(bool parallel, void (*forEachChunk)(lingodb::runtime::RecordBatchInfo*, void*), void* context) {
   utility::Tracer::Trace trace(tableScan);
   dataSource->iterate(parallel, members, [context, forEachChunk](lingodb::runtime::RecordBatchInfo* recordBatchInfo) {
      forEachChunk(recordBatchInfo, context);
   });
   trace.stop();
}