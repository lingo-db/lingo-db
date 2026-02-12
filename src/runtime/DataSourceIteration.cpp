#include "lingodb/runtime/DataSourceIteration.h"
//TODO remove
#include "../../include/lingodb/runtime/DatasourceRestrictionProperty.h"
#include "json.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"
#include <iterator>

#include "lingodb/utility/Tracer.h"

#include <arrow/array.h>
#include <arrow/table.h>
namespace utility = lingodb::utility;
namespace {

static utility::Tracer::Event tableScan("Tablescan", "tableScan");

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
   //TODO remove init
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
   std::string tableName;
   std::unordered_map<std::string, std::string> memberToColumn;
   std::unordered_set<FilterDescription> uniqueRestrictions;
   std::vector<FilterDescription> restrictions;

   std::string dataSourceRaw = description.str();
   auto dataSource = utility::deserializeFromHexString<ExternalDatasourceProperty>(dataSourceRaw);
   tableName = dataSource.tableName;
   for (auto& filterDesc : dataSource.filterDescriptions) {
      if (uniqueRestrictions.contains(filterDesc)) {
         continue;
      }
      uniqueRestrictions.insert(filterDesc);
      restrictions.push_back(filterDesc);
   }
   for (auto& mapping : dataSource.mapping) {
      memberToColumn[mapping.memberName] = mapping.identifier;
   }
   auto& session = executionContext->getSession();
   if (auto maybeRelation = session.getCatalog()->getTypedEntry<catalog::TableCatalogEntry>(tableName)) {
      auto relation = maybeRelation.value();

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