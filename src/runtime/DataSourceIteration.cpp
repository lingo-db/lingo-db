#include "lingodb/runtime/DataSourceIteration.h"
//TODO remove
#include "../../include/lingodb/runtime/DatasourceRestrictionProperty.h"
#include "json.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"
#include <iterator>

#include "lingodb/utility/Tracer.h"

#include <arrow/array.h>
#include <arrow/table.h>
#include <lingodb/runtime/storage/LingoDBTable.h>
namespace utility = lingodb::utility;
namespace {

static utility::Tracer::Event tableScan("Tablescan", "tableScan");
static utility::Tracer::Event tableScanGPU("TablescanGPU", "tableScan");
constexpr uint32_t rowLimitMultibatch{100'000'000};

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
   void iterateGPU(bool parallel, bool gpu, std::vector<std::string> members, const std::function<void(lingodb::runtime::BatchView*, int64_t, int64_t, int64_t, void*)>& cb) override {
      namespace runtime = lingodb::runtime;
      auto* executionContext = runtime::getCurrentExecutionContext();
      std::cout << "Running over ... batches, parallel=" << parallel << ", gpu=" << gpu << "\n";
      assert(&executionContext->getGPUMemManager());
      auto& gpuManager = executionContext->getGPUMemManager();
      lingodb::runtime::LingoDBTable* table = dynamic_cast<lingodb::runtime::LingoDBTable*>(&tableStorage);
      auto& tableData = table->rawData();
      std::vector<size_t> colIds;
      for (auto member : members) {
         auto colId = table->getColIndex(memberToColumn.at(member));
         if (colId == -1) {
            throw std::runtime_error("Column not found");
         }
         colIds.push_back(colId);
      }

      auto* dummyValidData = executionContext->getGPUMemManager().fetchArray(lingodb::runtime::ArrayView::validData.data(), lingodb::runtime::ArrayView::validData.size());

      std::vector<const lingodb::runtime::LingoDBTable::TableChunk*> tableChunksToLaunch;
      std::vector<uint8_t*> buffers;
      buffers.reserve(colIds.size() * 5);

      const int64_t smCount{static_cast<int64_t>(gpuManager.getSMCount(0))};
      auto launchMultiBatch = [&]() {
         assert(tableChunksToLaunch.size() && "A launch must contain >0 batches!");
         buffers.clear();

         for (const auto* chunk : tableChunksToLaunch) {
            for (auto col : colIds) {
               auto arrayData = chunk->data()->column_data(col);
               auto idx = 0;
               for (const auto& buf : arrayData->buffers) {
                  if (buf) {
                     buffers.push_back(executionContext->getGPUMemManager().fetchArray(reinterpret_cast<uint8_t*>(buf->address()), buf->size()));
                  } else {
                     if (idx == 0) {
                        // valid buffer, use dummy
                        buffers.push_back(dummyValidData);
                     } else {
                        // data or varlen buffer, use nullptr
                        buffers.push_back(nullptr);
                     }
                     idx++;
                  }
               }
            }
         }
         std::vector<lingodb::runtime::ArrayView> arrayViews(colIds.size() * tableChunksToLaunch.size());
         uint8_t** buffersGPUPtr = reinterpret_cast<uint8_t**>(gpuManager.getPtrForArray(buffers.size() * sizeof(uint8_t*)));
         gpuManager.moveToDevice(reinterpret_cast<uint8_t*>(buffers.data()), reinterpret_cast<uint8_t*>(buffersGPUPtr), buffers.size() * sizeof(uint8_t*), true);
         size_t currBufferOffset = 0;
         size_t currColIdOffset = 0;
         for (const auto& chunk : tableChunksToLaunch) {
            for (auto colId : colIds) {
               arrayViews[currColIdOffset] = lingodb::runtime::ArrayView{
                  .length = chunk->data()->column_data(colId)->length,
                  .nullCount = chunk->data()->column_data(colId)->null_count,
                  .offset = chunk->data()->column_data(colId)->offset,
                  .nBuffers = static_cast<int64_t>(chunk->data()->column_data(colId)->buffers.size()),
                  .nChildren = static_cast<int64_t>(chunk->data()->column_data(colId)->child_data.size()),
                  .buffers = (const void**) &buffersGPUPtr[currBufferOffset],
                  .children = nullptr};
               currBufferOffset += chunk->data()->column_data(colId)->buffers.size();
               currColIdOffset++;
            }
         }
         runtime::ArrayView* arrayViewsGPU = reinterpret_cast<runtime::ArrayView*>(gpuManager.getPtrForArray(arrayViews.size() * sizeof(runtime::ArrayView)));
         gpuManager.moveToDevice(
            reinterpret_cast<uint8_t*>(arrayViews.data()),
            reinterpret_cast<uint8_t*>(arrayViewsGPU),
            arrayViews.size() * sizeof(runtime::ArrayView),
            true);

         const runtime::ArrayView** arrayViewsGpuPtr = reinterpret_cast<const runtime::ArrayView**>(gpuManager.getPtrForArray(arrayViews.size() * sizeof(runtime::ArrayView*)));
         std::vector<runtime::ArrayView*> arrayViewsPtr(arrayViews.size());
         for (size_t i = 0; i < arrayViews.size(); i++) {
            arrayViewsPtr[i] = &arrayViewsGPU[i];
         }
         gpuManager.moveToDevice(
            reinterpret_cast<uint8_t*>(arrayViewsPtr.data()),
            reinterpret_cast<uint8_t*>(arrayViewsGpuPtr),
            arrayViews.size() * sizeof(runtime::ArrayView*),
            true);
         std::vector<runtime::BatchView> batchViews(tableChunksToLaunch.size());
         for (size_t i = 0; i < tableChunksToLaunch.size(); i++) {
            auto& batchView = batchViews[i];
            batchView.length = tableChunksToLaunch[i]->data()->num_rows();
            batchView.offset = 0;
            batchView.arrays = &arrayViewsGpuPtr[i * colIds.size()];
            batchView.selectionVector = nullptr; // currently not used, but could be set to defaultSelectionVector if needed
         }
         runtime::BatchView* batchViewsGpuPtr = reinterpret_cast<runtime::BatchView*>(gpuManager.getPtrForArray(batchViews.size() * sizeof(runtime::BatchView)));
         gpuManager.moveToDevice(
            reinterpret_cast<uint8_t*>(batchViews.data()),
            reinterpret_cast<uint8_t*>(batchViewsGpuPtr),
            batchViews.size() * sizeof(runtime::BatchView),
            true);
         int64_t numBatchesInSet = batchViews.size();
         int64_t gridDimX{std::min(numBatchesInSet, smCount)};
         int64_t blockDimX{256};
         cb(batchViewsGpuPtr, numBatchesInSet, gridDimX, blockDimX, gpuManager.getThreadStream()); // currently assume implicit stream sync
         gpuManager.freeArray((uint8_t*) buffersGPUPtr);
         gpuManager.freeArray((uint8_t*) arrayViewsGPU);
         gpuManager.freeArray((uint8_t*) arrayViewsGpuPtr);
         gpuManager.freeArray((uint8_t*) batchViewsGpuPtr);
      };
      //group chunks and then launch in group
      int64_t numRowsPerLaunch{0};
      for (const auto& batch : tableData) {
         if (numRowsPerLaunch > rowLimitMultibatch) {
            launchMultiBatch();
            tableChunksToLaunch.clear();
            numRowsPerLaunch = 0;
         } else {
            tableChunksToLaunch.push_back(&batch);
            numRowsPerLaunch += batch.data()->num_rows();
         }
      }
      if (!tableChunksToLaunch.empty()) {
         //launch last batch
         launchMultiBatch();
      }
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

void lingodb::runtime::DataSourceIteration::iterateGPU(void (*forEachChunk)(runtime::BatchView*, int64_t, void*, int64_t, int64_t, void*), void* context) {
   utility::Tracer::Trace trace(tableScanGPU);
   dataSource->iterateGPU(false, true, members, [context, forEachChunk](runtime::BatchView* recordBatchInfo, int64_t numBatches, uint64_t gridDimX, uint64_t blockDimX, void* stream) {
      forEachChunk(recordBatchInfo, numBatches, context, gridDimX, blockDimX, stream);
   });
   trace.stop();
}
