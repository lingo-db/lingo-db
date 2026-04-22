#include "lingodb/runtime/storage/ExternalTableScan.h"

#include <arrow/record_batch.h>

#include <iostream>
#include <shared_mutex>

#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/statistics.h>

#include <arrow/util/decimal.h>

#include <cstring>
#include <optional>
namespace lingodb::runtime {

namespace {
std::optional<int64_t> decimalFilterValueToInt64(const std::variant<std::string, int64_t, double>& filterValue, int32_t targetScale) {
   __int128 decimalValue = 0;
   if (std::holds_alternative<std::string>(filterValue)) {
      int32_t parsedPrecision;
      int32_t parsedScale;
      arrow::Decimal128 parsed;
      if (!arrow::Decimal128::FromString(std::get<std::string>(filterValue), &parsed, &parsedPrecision, &parsedScale).ok()) {
         return std::nullopt;
      }
      auto rescaled = parsed.Rescale(parsedScale, targetScale);
      if (!rescaled.ok()) {
         return std::nullopt;
      }
      std::memcpy(&decimalValue, rescaled.ValueUnsafe().native_endian_bytes(), sizeof(decimalValue));
   } else if (std::holds_alternative<int64_t>(filterValue)) {
      decimalValue = static_cast<__int128>(std::get<int64_t>(filterValue));
      int32_t scale = targetScale;
      while (scale > 0) {
         if (decimalValue > std::numeric_limits<int64_t>::max() / 10 || decimalValue < std::numeric_limits<int64_t>::min() / 10) {
            return std::nullopt;
         }
         decimalValue *= 10;
         scale--;
      }
   } else {
      return std::nullopt;
   }

   if (decimalValue > std::numeric_limits<int64_t>::max() || decimalValue < std::numeric_limits<int64_t>::min()) {
      return std::nullopt;
   }
   return static_cast<int64_t>(decimalValue);
}
} // namespace
template <class T>
bool MinMaxMetadataFilter<T>::apply(std::shared_ptr<parquet::Statistics> stats) const {
   if (!stats || !stats->HasMinMax()) {
      return true;
   }
   bool comparable = true;
   T rowGroupMin{};
   T rowGroupMax{};
   if constexpr (std::is_integral_v<T>) {
      switch (stats->physical_type()) {
         case parquet::Type::INT32: {
            auto intStats = std::dynamic_pointer_cast<parquet::Int32Statistics>(stats);
            if (!intStats) {
               comparable = false;
               break;
            }
            rowGroupMin = static_cast<T>(intStats->min());
            rowGroupMax = static_cast<T>(intStats->max());
            break;
         }
         case parquet::Type::INT64: {
            auto intStats = std::dynamic_pointer_cast<parquet::Int64Statistics>(stats);
            if (!intStats) {
               comparable = false;
               break;
            }
            rowGroupMin = static_cast<T>(intStats->min());
            rowGroupMax = static_cast<T>(intStats->max());
            break;
         }
         default:
            comparable = false;
            break;
      }
   } else if constexpr (std::is_same_v<T, std::string>) {
      if (stats->physical_type() != parquet::Type::BYTE_ARRAY) {
         comparable = false;
      } else {
         auto strStats = std::dynamic_pointer_cast<parquet::ByteArrayStatistics>(stats);
         if (!strStats) {
            comparable = false;
         } else {
            const auto& minVal = strStats->min();
            const auto& maxVal = strStats->max();
            rowGroupMin = std::string(reinterpret_cast<const char*>(minVal.ptr), minVal.len);
            rowGroupMax = std::string(reinterpret_cast<const char*>(maxVal.ptr), maxVal.len);
         }
      }
   } else {
      comparable = false;
   }
   if (!comparable) {
      return true;
   }
   if (min.has_value() && (minInclusive ? (rowGroupMax < min) : (rowGroupMax <= min))) {
      return false;
   }
   if (max.has_value() && (maxInclusive ? (rowGroupMin > max) : (rowGroupMin >= max))) {
      return false;
   }

   return true;
}
bool NotNullFilter::apply(std::shared_ptr<parquet::Statistics> stats) const {
   return stats->num_values() > 0;
}
bool NullFilter::apply(std::shared_ptr<parquet::Statistics> stats) const {
   if (!stats || !stats->HasNullCount()) {
      return true;
   }
   return stats->null_count() > 0;
}

bool ParquetBatchesWorkerResvState::rowGroupPassesMetadataFilters(int rowGroup, std::unique_ptr<parquet::arrow::FileReader>& localReader) const {
   if (metadataFilters.empty()) {
      return true;
   }

   parquet::ParquetFileReader* parquetReader = localReader->parquet_reader();
   if (!parquetReader) {
      return true;
   }

   std::shared_ptr<parquet::FileMetaData> fileMetadata = parquetReader->metadata();
   if (!fileMetadata || rowGroup < 0 || rowGroup >= fileMetadata->num_row_groups()) {
      return true;
   }

   std::unique_ptr<parquet::RowGroupMetaData> rowGroupMetadata = fileMetadata->RowGroup(rowGroup);
   if (!rowGroupMetadata) {
      return true;
   }

   for (const auto& [columnId, filter] : metadataFilters) {
      if (columnId < 0 || columnId >= rowGroupMetadata->num_columns()) {
         continue;
      }

      std::unique_ptr<parquet::ColumnChunkMetaData> columnMetadata = rowGroupMetadata->ColumnChunk(columnId);
      if (!columnMetadata || !columnMetadata->is_stats_set()) {
         continue;
      }

      std::shared_ptr<parquet::Statistics> stats = columnMetadata->statistics();
      if (!filter || !filter->apply(stats)) {
         std::cerr << "Filter\n";
         return false;
      }
   }

   return true;
}

int ParquetBatchesWorkerResvState::fetchNextMatchingRowGroup(std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::unique_ptr<parquet::arrow::FileReader>& localReader) const {
   while (true) {
      int nextRgId = rgIdstartIndex.fetch_add(1);
      if (nextRgId >= numberOfRowGroups) {
         return -1;
      }

      if (rowGroupPassesMetadataFilters(nextRgId, localReader)) {
         return nextRgId;
      }
   }
}

bool ParquetBatchesWorkerResvState::hasMoreWork() {
   std::shared_lock<std::shared_mutex> stateLock(this->mutex);
   return resvCursor < unitAmount;
}
void ParquetBatchesWorkerResvState::initNewRowGroup(int rowGroup, size_t splitSize, std::vector<std::deque<std::shared_ptr<ChunkWorkEntry>>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader) {
   std::vector<int> newRgIds = {rowGroup};
   auto newlocalRowGroupReaderUncertain = localReader->GetRecordBatchReader(newRgIds);
   if (!newlocalRowGroupReaderUncertain.ok()) {
      //TODO handle error
      std::cerr << "Should not happen" << std::endl;
   }
   std::swap(rowGroupRecordBatchReader, newlocalRowGroupReaderUncertain.ValueOrDie());
   rgId = rowGroup;

   resvId = 0;

   if (prefetchedRgId == -1) {
      prefetchedRgId = fetchNextMatchingRowGroup(rgIdstartIndex, numberOfRowGroups, localReader);
   }

   std::vector<int> bufferRgIds = {rowGroup};
   if (prefetchedRgId != -1) {
      bufferRgIds.push_back(prefetchedRgId);
   }

   localReader->parquet_reader()->PreBuffer(bufferRgIds, colIds, arrow::io::default_io_context(), arrow::io::CacheOptions::Defaults());
   arrow::Future<> buffering = localReader->parquet_reader()->WhenBuffered(newRgIds, colIds);
   auto* queryLifetimeChunksPtr = queryLifetimeChunks;
   auto* rgIdstartIndexPtr = &rgIdstartIndex;
   auto* colIdsPtr = &colIds;
   auto* localReaderPtr = &localReader;
   const size_t capturedSplitSize = splitSize;
   const int capturedNumberOfRowGroups = numberOfRowGroups;
   const int capturedRowGroup = rowGroup;
   buffering.AddCallback([this, queryLifetimeChunksPtr, rgIdstartIndexPtr, capturedNumberOfRowGroups, colIdsPtr, localReaderPtr, capturedSplitSize, capturedRowGroup](const arrow::Status& status) {
      if (!status.ok()) {
         std::cerr << "PreBuffer failed for row group " << capturedRowGroup << ": " << status.ToString() << std::endl;
         isBuffering = false;
         return;
      }

      auto workInfo = tryFetchNextRecordBatch(capturedSplitSize, queryLifetimeChunksPtr, *rgIdstartIndexPtr, capturedNumberOfRowGroups, *colIdsPtr, *localReaderPtr);
      assert(workInfo.currentMorsel != -1);
      //Found new work
      resvId = workInfo.currentMorsel;
      reservedChunkId = workInfo.ownChunkId;
      isBuffering = false;
   });
}
ParquetBatchesWorkerResvState::WorkInfo ParquetBatchesWorkerResvState::fetchAndNextOwn(size_t splitSize, std::vector<std::deque<std::shared_ptr<ChunkWorkEntry>>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader) {
   std::unique_lock<std::shared_mutex> stateLock(this->mutex);

   auto workerId = lingodb::scheduler::currentWorkerId();
   long curr = resvCursor++;
   //Has more work localy on the "current" TableChunk!
   if (curr < unitAmount) {
      return {.ownChunkId = ownChunkId, .currentMorsel = curr};
   }
   if (!rowGroupRecordBatchReader) {
      int newRgId;
      if (prefetchedRgId != -1) {
         newRgId = prefetchedRgId;
         prefetchedRgId = -1;
      } else {
         newRgId = fetchNextMatchingRowGroup(rgIdstartIndex, numberOfRowGroups, localReader);
      }
      if (newRgId < 0 || newRgId >= numberOfRowGroups) {
         //No (new) rowgroup to work on
         fullyExhausted.store(true, std::memory_order_release);
         return {.ownChunkId = ownChunkId, .currentMorsel = -1};
      }
      isBuffering = true;
      initNewRowGroup(newRgId, splitSize, queryLifetimeChunks, rgIdstartIndex, numberOfRowGroups, colIds, localReader);
      return {.buffering = true};
   }
   return tryFetchNextRecordBatch(splitSize, queryLifetimeChunks, rgIdstartIndex, numberOfRowGroups, colIds, localReader);
}

ParquetBatchesWorkerResvState::WorkInfo ParquetBatchesWorkerResvState::tryFetchNextRecordBatch(size_t splitSize, std::vector<std::deque<std::shared_ptr<ChunkWorkEntry>>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader) {
   std::shared_ptr<arrow::RecordBatch> batch;
   //If correctly buffered this call should not block
   auto status = rowGroupRecordBatchReader->ReadNext(&batch);
   if (!status.ok()) {
      std::cerr << "Error: Should not happen " << std::endl;
   }
   if (batch) {
      //new work found for current rowGroup
      auto& localChunks = (*queryLifetimeChunks)[workerId];
      //New Chunk
      auto entry = std::make_shared<ChunkWorkEntry>();
      entry->chunk = std::make_shared<LingoDBTable::TableChunk>(batch, 0);
      entry->totalMorsels = (entry->chunk->getNumRows() + splitSize - 1) / splitSize;
      localChunks.push_back(entry);
      ownChunkId = localChunks.size() - 1;
      unitAmount = entry->totalMorsels;
      resvCursor = 1;
      resvId = 0;

      return {.ownChunkId = ownChunkId, .currentMorsel = 0};
   } else {
      //No additional work found for current rowGroup, try to increase rowGroup to find new work
      int newRgId;
      if (prefetchedRgId != -1) {
         newRgId = prefetchedRgId;
         prefetchedRgId = -1;
      } else {
         newRgId = fetchNextMatchingRowGroup(rgIdstartIndex, numberOfRowGroups, localReader);
      }
      if (newRgId < 0 || newRgId >= numberOfRowGroups) {
         //No work found
         fullyExhausted.store(true, std::memory_order_release);
         return {.ownChunkId = ownChunkId, .currentMorsel = -1};
      }
      isBuffering = true;
      initNewRowGroup(newRgId, splitSize, queryLifetimeChunks, rgIdstartIndex, numberOfRowGroups, colIds, localReader);
      return {.buffering = true};
   }
}

std::pair<size_t, int> ParquetBatchesWorkerResvState::fetchAndNext() {
   std::unique_lock<std::shared_mutex> resvLock(mutex);
   long cur = resvCursor++;
   return {ownChunkId, cur >= unitAmount ? -1 : cur};
}

//------------------------------------------------------
ScanParquetFileTask::ScanParquetFileTask(std::string filePath, std::vector<int> colids, std::function<void(BatchView*)> cb, std::unique_ptr<Restrictions> restrictions, std::vector<FilterDescription> filterDescriptions) : filePath(std::move(filePath)), cb(cb), restrictions(std::move(restrictions)), colIds(std::move(colids)) {
   auto status = init(filterDescriptions);
   if (!status.ok()) {
      std::cerr << "Error while loading ScanParquetFileTask: Should not happen " << std::endl;
      std::cerr << status.ToString() << std::endl;
   }
}
arrow::Status ScanParquetFileTask::init(std::vector<FilterDescription>& filterDescriptions) {
   const size_t numWorkers = lingodb::scheduler::getNumWorkers();
   queryLifetimeChunks = new std::vector<std::deque<std::shared_ptr<ParquetBatchesWorkerResvState::ChunkWorkEntry>>>(numWorkers);
   //Keep in memory until query is finished
   //Joins for instance do not copy data and only refernces them and therefore the lifetime of the TableChunks has to exceed this table scan
   /* lingodb::runtime::getCurrentExecutionContext()->registerState({queryLifetimeChunks, [](void* ptr) {
                                                                     delete reinterpret_cast<std::vector<std::deque<LingoDBTable::TableChunk>>*>(ptr);
                                                                  }});*/
   std::shared_ptr<arrow::io::RandomAccessFile> input;
   ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(filePath));
   std::shared_ptr<arrow::Schema> parquetSchema;
   {
      std::unique_ptr<parquet::arrow::FileReader> schemaReader;
      ARROW_ASSIGN_OR_RAISE(schemaReader, parquet::arrow::OpenFile(input, arrow::default_memory_pool()));
      ARROW_RETURN_NOT_OK(schemaReader->GetSchema(&parquetSchema));
   }

   for (const auto& filter : filterDescriptions) {
      int metadataColumnId = parquetSchema->GetFieldIndex(filter.columnName);
      if (metadataColumnId < 0) {
         continue;
      }
      if (filter.op == FilterOp::NOTNULL) {
         metadataFilters.emplace_back(metadataColumnId, std::make_shared<NotNullFilter>());
         continue;
      }
      auto field = parquetSchema->field(metadataColumnId);
      if (!field) {
         continue;
      }
      auto typeId = field->type()->id();
      std::variant<std::string, int64_t, double> comparableValue;
      bool hasComparableValue = false;
      if (typeId == arrow::Type::INT32 || typeId == arrow::Type::INT64) {
         if (!std::holds_alternative<int64_t>(filter.value)) {
            continue;
         }
         comparableValue = std::get<int64_t>(filter.value);
         hasComparableValue = true;
      } else if (typeId == arrow::Type::DECIMAL32 || typeId == arrow::Type::DECIMAL64 || typeId == arrow::Type::DECIMAL128) {
         auto decimalType = std::dynamic_pointer_cast<arrow::DecimalType>(field->type());
         if (!decimalType) {
            continue;
         }
         auto converted = decimalFilterValueToInt64(filter.value, decimalType->scale());
         if (converted.has_value()) {
            comparableValue = *converted;
            hasComparableValue = true;
         }
      } else if (typeId == arrow::Type::STRING || typeId == arrow::Type::LARGE_STRING) {
         if (!std::holds_alternative<std::string>(filter.value)) {
            continue;
         }
         comparableValue = std::get<std::string>(filter.value);
         hasComparableValue = true;
      }
      if (!hasComparableValue) {
         continue;
      }
      std::visit(
         [&](const auto& value) {
            using ValueType = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<ValueType, double>) {
               return;
            } else {
               switch (filter.op) {
                  case FilterOp::EQ:
                     metadataFilters.emplace_back(metadataColumnId, std::make_shared<MinMaxMetadataFilter<ValueType>>(value, value, true, true));
                     break;
                  case FilterOp::GT:
                     metadataFilters.emplace_back(metadataColumnId, std::make_shared<MinMaxMetadataFilter<ValueType>>(value, std::nullopt, false));
                     break;
                  case FilterOp::GTE:
                     metadataFilters.emplace_back(metadataColumnId, std::make_shared<MinMaxMetadataFilter<ValueType>>(value, std::nullopt, true));
                     break;
                  case FilterOp::LT:
                     metadataFilters.emplace_back(metadataColumnId, std::make_shared<MinMaxMetadataFilter<ValueType>>(std::nullopt, value, false, false));
                     break;
                  case FilterOp::LTE:
                     metadataFilters.emplace_back(metadataColumnId, std::make_shared<MinMaxMetadataFilter<ValueType>>(std::nullopt, value, false, true));
                     break;
                  default:
                     break;
               }
            }
         },
         comparableValue);
   }

   {
      auto metadataReader = parquet::ParquetFileReader::Open(input);
      numOfRowGroups = metadataReader->metadata()->num_row_groups();
   }

   localRowGroup.assign(numWorkers, std::numeric_limits<int64_t>::max());
   arrayViewPtrs.assign(numWorkers, std::vector<const ArrayView*>(colIds.size()));
   batchInfos.resize(numWorkers);
   selVecs.resize(numWorkers);
   readers.resize(numWorkers);
   workerResvs.resize(numWorkers);

   for (size_t i = 0; i < numWorkers; i++) {
      batchInfos[i].arrays = arrayViewPtrs[i].data();
      selVecs[i] = std::make_pair(new uint16_t[splitSize], new uint16_t[splitSize]);

      auto parquetFileReader = parquet::ParquetFileReader::Open(input);

      std::unique_ptr<parquet::arrow::FileReader> reader;

      ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(input, arrow::default_memory_pool()));
      readers[i] = std::move(reader);
      workerResvs[i] = std::make_unique<ParquetBatchesWorkerResvState>();
      workerResvs[i]->metadataFilters = metadataFilters;
      workerResvs[i]->workerId = i;
   }

   return arrow::Status::OK();
}
bool ScanParquetFileTask::allocateWork() {
   // quick check for exhaust. workExhausted is true if there is no more buffer or no more
   // work unit in own local state or steal from other workers.
   if (workExhausted) {
      return false;
   }
   //1. if the current worker has more work locally, do it
   auto* state = workerResvs[lingodb::scheduler::currentWorkerId()].get();
   auto workInfo = state->fetchAndNextOwn(splitSize, queryLifetimeChunks, nextRowGroup, numOfRowGroups, colIds, readers[lingodb::scheduler::currentWorkerId()]);
   if (workInfo.buffering) {
      return true;
   }
   if (workInfo.currentMorsel != -1) {
      //Found new work
      state->resvId = workInfo.currentMorsel;
      state->reservedChunkId = workInfo.ownChunkId;
      return true;
   }

   //3. if the current worker has no more work locally and no more work globally, try to steal work from the worker we stole from last time
   if (state->stealWorkerId != std::numeric_limits<size_t>::max()) {
      auto* other = workerResvs[state->stealWorkerId].get();
      if (other->hasMoreWork()) {
         auto [otherChunkId, id] = other->fetchAndNext();
         if (id != -1) {
            state->resvId = id;
            state->reservedChunkId = otherChunkId;
            return true;
         }
      }
      state->stealWorkerId = std::numeric_limits<size_t>::max();
   }

   //4. if the current worker has no more work locally and no more work globally, try to steal work from other workers
   for (size_t i = 1; i < workerResvs.size(); i++) {
      // make sure index of worker to steal never exceed worker number limits
      auto idx = (lingodb::scheduler::currentWorkerId() + i) % workerResvs.size();
      auto* other = workerResvs[idx].get();
      if (other->hasMoreWork()) {
         auto [otherChunkId, id] = other->fetchAndNext();
         if (id != -1) {
            // only current worker can modify its own stealWorkerId. no need to lock
            state->stealWorkerId = idx;
            state->resvId = id;
            state->reservedChunkId = otherChunkId;
            return true;
         }
      }
   }

   // No work found right now. Only mark the whole task as exhausted once all workers
   // reported that they can never produce more work.
   bool allFullyExhausted = true;
   for (auto& s : workerResvs) {
      if (!s->fullyExhausted.load(std::memory_order_acquire)) {
         allFullyExhausted = false;
         break;
      }
   }
   if (allFullyExhausted) {
      workExhausted.store(true);
   }

   return false;
}
void ScanParquetFileTask::unitRun(LingoDBTable::TableChunk& chunk) {
   auto workerId = lingodb::scheduler::currentWorkerId();
   auto* state = workerResvs[workerId].get();
   lingodb::runtime::BatchView& batchView = batchInfos[workerId];
   auto [selVec1, selVec2] = selVecs[workerId];

   size_t begin = splitSize * state->resvId;
   size_t len = std::min(begin + splitSize, chunk.getNumRows()) - begin;
   batchView.offset = begin;
   batchView.selectionVector = BatchView::defaultSelectionVector.data();
   batchView.length = std::min(static_cast<size_t>(chunk.getNumRows() - begin), len);
   //TODO trace
   for (size_t i = 0; i < colIds.size(); i++) {
      batchView.arrays[i] = chunk.getArrayView(colIds[i]);
   }
   auto [newLen, selVec] = restrictions->applyFilters(begin, batchView.length, selVec1, selVec2, [&](size_t colId) {
      return chunk.getArrayView(colId);
   });
   batchView.length = newLen;
   batchView.selectionVector = selVec;
   if (batchView.length > 0) {
      cb(&batchView);
   }
}
void ScanParquetFileTask::performWork() {
   auto workerId = lingodb::scheduler::currentWorkerId();
   auto* state = workerResvs[workerId].get();
   if (state->stealWorkerId == std::numeric_limits<size_t>::max()) {
      while (state->isBuffering == true) {
         lingodb::scheduler::yieldCurrentTask();
      }
   }

   const size_t ownerWorkerId = state->stealWorkerId == std::numeric_limits<size_t>::max() ? workerId : state->stealWorkerId;
   auto* ownerState = workerResvs[ownerWorkerId].get();
   auto& chunks = (*queryLifetimeChunks)[ownerWorkerId];

   std::shared_ptr<ParquetBatchesWorkerResvState::ChunkWorkEntry> entry;
   {
      std::shared_lock<std::shared_mutex> ownerLock(ownerState->mutex);
      if (state->reservedChunkId >= chunks.size()) {
         return;
      }
      entry = chunks[state->reservedChunkId];
   }
   if (!entry || !entry->chunk) {
      return;
   }

   unitRun(*entry->chunk);

#if !KEEP_IN_MEMEORY
   long finishedMorsels = entry->completedMorsels.fetch_add(1, std::memory_order_acq_rel) + 1;
   if (finishedMorsels >= entry->totalMorsels) {
      std::unique_lock<std::shared_mutex> ownerLock(ownerState->mutex);
      if (state->reservedChunkId < chunks.size() && chunks[state->reservedChunkId] == entry) {
         chunks[state->reservedChunkId].reset();
      }
   }
#endif
}
ScanParquetFileTask::~ScanParquetFileTask() {
   for (auto& [selVec1, selVec2] : selVecs) {
      delete[] selVec1;
      delete[] selVec2;
   }
#if !KEEP_IN_MEMEORY
   delete queryLifetimeChunks;
#endif
}

} // namespace lingodb::runtime