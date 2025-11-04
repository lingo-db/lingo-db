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
   std::vector<lingodb::runtime::FilterDescription> restrictions;

   public:
   TableSource(lingodb::runtime::TableStorage& tableStorage, std::unordered_map<std::string, std::string> memberToColumn, std::vector<lingodb::runtime::FilterDescription> restrictions) : tableStorage(tableStorage), memberToColumn(memberToColumn), restrictions(restrictions) {}
   void iterate(bool parallel, std::vector<std::string> members, const std::function<void(lingodb::runtime::BatchView*)>& cb) override {
      std::vector<std::string> columns;
      for (const auto& member : members) {
         columns.push_back(memberToColumn.at(member));
      }
      auto scanTask = tableStorage.createScanTask({parallel, columns, restrictions, cb});
      lingodb::scheduler::awaitChildTask(std::move(scanTask));
   }
};
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
   auto* it = new DataSourceIteration(dataSource, members);
   getCurrentExecutionContext()->registerState({it, [](void* ptr) { delete reinterpret_cast<DataSourceIteration*>(ptr); }});
   return it;
}
lingodb::runtime::DataSourceIteration::DataSourceIteration(DataSource* dataSource, const std::vector<std::string>& members) : dataSource(dataSource), members(members) {
}

lingodb::runtime::DataSource* lingodb::runtime::DataSource::get(lingodb::runtime::VarLen32 description) {
   lingodb::runtime::ExecutionContext* executionContext = lingodb::runtime::getCurrentExecutionContext();
   nlohmann::json descr = nlohmann::json::parse(description.str());
   std::string tableName = descr["table"];
   std::vector<FilterDescription> restrictions;
   if (descr.contains("restrictions")) {
      for (auto r : descr["restrictions"].get<nlohmann::json::array_t>()) {
         FilterDescription filterDesc;
         filterDesc.columnName = r["column"].get<std::string>();
         std::string op = r["cmp"].get<std::string>();
         if (op == "lt") {
            filterDesc.op = FilterOp::LT;
         } else if (op == "lte") {
            filterDesc.op = FilterOp::LTE;
         } else if (op == "gt") {
            filterDesc.op = FilterOp::GT;
         } else if (op == "gte") {
            filterDesc.op = FilterOp::GTE;
         } else if (op == "eq") {
            filterDesc.op = FilterOp::EQ;
         } else if (op == "neq") {
            filterDesc.op = FilterOp::NEQ;
         } else if (op == "isnotnull") {
            filterDesc.op = FilterOp::NOTNULL;
         } else {
            throw std::runtime_error("unsupported filter op");
         }
         if (r["value"].is_string()) {
            filterDesc.value = r["value"].get<std::string>();
         } else if (r["value"].is_number_integer()) {
            filterDesc.value = r["value"].get<int64_t>();
         } else if (r["value"].is_number_float()) {
            filterDesc.value = r["value"].get<double>();
         } else {
            throw std::runtime_error("unsupported filter value type");
         }
         restrictions.push_back(filterDesc);
      }
   }
   auto& session = executionContext->getSession();
   if (auto maybeRelation = session.getCatalog()->getTypedEntry<catalog::TableCatalogEntry>(tableName)) {
      auto relation = maybeRelation.value();
      std::unordered_map<std::string, std::string> memberToColumn;
      for (auto m : descr["mapping"].get<nlohmann::json::object_t>()) {
         memberToColumn[m.first] = m.second.get<std::string>();
      }
      auto* ts = new TableSource(relation->getTableStorage(), memberToColumn, restrictions);
      getCurrentExecutionContext()->registerState({ts, [](void* ptr) { delete reinterpret_cast<TableSource*>(ptr); }});
      return ts;

   } else {
      throw std::runtime_error("could not find relation");
   }
}

void lingodb::runtime::DataSourceIteration::iterate(bool parallel, void (*forEachChunk)(lingodb::runtime::BatchView*, void*), void* context) {
   utility::Tracer::Trace trace(tableScan);
   dataSource->iterate(parallel, members, [context, forEachChunk](lingodb::runtime::BatchView* recordBatchInfo) {
      forEachChunk(recordBatchInfo, context);
   });
   trace.stop();
}