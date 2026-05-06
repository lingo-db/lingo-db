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
#include <arrow/record_batch.h>
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

// Wraps an in-memory arrow::Table (e.g., produced by a tabular Python UDF)
// as a DataSource. Iterates serially over its record batches and feeds each
// as a BatchView to the supplied callback. Members are resolved by Arrow
// column name — the local_table type carries column names that match the
// Arrow schema produced by the UDF.
class ArrowTableSource : public lingodb::runtime::DataSource {
   std::shared_ptr<arrow::Table> table;
   std::vector<std::vector<const void*>> bufferOwner;
   std::vector<std::vector<lingodb::runtime::ArrayView>> arrayViewOwner;
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;

   void buildBatchView(size_t batchIdx, const std::vector<int>& colIdxs,
                       std::vector<const lingodb::runtime::ArrayView*>& arrayPtrs,
                       lingodb::runtime::BatchView& view) {
      auto& batch = batches[batchIdx];
      auto& bufVec = bufferOwner[batchIdx];
      auto& views = arrayViewOwner[batchIdx];
      // Lazy build per-batch ArrayViews.
      if (views.empty()) {
         std::vector<size_t> bufferStart;
         for (int colId = 0; colId < batch->num_columns(); colId++) {
            auto data = batch->column(colId)->data();
            size_t startBuf = bufVec.size();
            for (auto& b : data->buffers) {
               bufVec.push_back(b ? b->data() : nullptr);
            }
            if (!bufVec[startBuf]) {
               if (data->length > lingodb::runtime::ArrayView::maxNullCount) {
                  throw std::runtime_error("ArrowTableSource: too many rows for null-bitmap fallback");
               }
               bufVec[startBuf] = lingodb::runtime::ArrayView::validData.data();
            }
            bufferStart.push_back(startBuf);
         }
         for (int colId = 0; colId < batch->num_columns(); colId++) {
            auto data = batch->column(colId)->data();
            views.push_back(lingodb::runtime::ArrayView{
               .length = data->length,
               .nullCount = data->null_count,
               .offset = data->offset,
               .nBuffers = static_cast<int64_t>(data->buffers.size()),
               .nChildren = static_cast<int64_t>(data->child_data.size()),
               .buffers = &bufVec[bufferStart[colId]],
               .children = nullptr});
         }
      }
      for (size_t i = 0; i < colIdxs.size(); i++) {
         arrayPtrs[i] = &views[colIdxs[i]];
      }
      view.length = batch->num_rows();
      view.offset = 0;
      view.selectionVector = lingodb::runtime::BatchView::defaultSelectionVector.data();
      view.arrays = arrayPtrs.data();
   }

   public:
   ArrowTableSource(std::shared_ptr<arrow::Table> table) : table(std::move(table)) {
      arrow::TableBatchReader reader(*this->table);
      std::shared_ptr<arrow::RecordBatch> batch;
      while (reader.ReadNext(&batch).ok() && batch) {
         batches.push_back(batch);
      }
      bufferOwner.resize(batches.size());
      arrayViewOwner.resize(batches.size());
   }
   void iterate(bool /*parallel*/, std::vector<std::string> members, const std::function<void(lingodb::runtime::BatchView*)>& cb) override {
      // Resolve member names to column indices in the underlying table.
      std::vector<int> colIdxs;
      colIdxs.reserve(members.size());
      for (const auto& m : members) {
         int idx = table->schema()->GetFieldIndex(m);
         if (idx < 0) {
            throw std::runtime_error("ArrowTableSource: unknown column '" + m + "' in tabular UDF result");
         }
         colIdxs.push_back(idx);
      }
      std::vector<const lingodb::runtime::ArrayView*> arrayPtrs(colIdxs.size());
      lingodb::runtime::BatchView view{};
      for (size_t i = 0; i < batches.size(); i++) {
         buildBatchView(i, colIdxs, arrayPtrs, view);
         cb(&view);
      }
   }
};
} // end namespace

lingodb::runtime::DataSource* lingodb::runtime::DataSource::fromArrowTable(lingodb::runtime::ArrowTable* table) {
   if (!table) {
      throw std::runtime_error("DataSource::fromArrowTable: null table");
   }
   auto* src = new ArrowTableSource(table->get());
   getCurrentExecutionContext()->registerState({src, [](void* ptr) { delete reinterpret_cast<ArrowTableSource*>(ptr); }});
   return src;
}

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