#include "lingodb/runtime/storage/ExternalTableScan.h"

#include <arrow/record_batch.h>

#include <iostream>
#include <shared_mutex>

#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
namespace lingodb::runtime {

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
      int nextRgId = rgIdstartIndex.fetch_add(1);
      if (nextRgId < numberOfRowGroups) {
         prefetchedRgId = nextRgId;
      }
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
         newRgId = rgIdstartIndex.fetch_add(1);
      }
      if (newRgId >= numberOfRowGroups) {
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
         newRgId = rgIdstartIndex.fetch_add(1);
      }
      if (newRgId >= numberOfRowGroups) {
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
   return {ownChunkId, cur >= unitAmount  ? -1 : cur};
}

//------------------------------------------------------
ScanParquetFileTask::ScanParquetFileTask(std::string filePath, std::vector<int> colids, std::function<void(BatchView*)> cb, std::unique_ptr<Restrictions> restrictions) : filePath(std::move(filePath)), cb(cb), restrictions(std::move(restrictions)), colIds(std::move(colids)) {
   auto status = init();
   if (!status.ok()) {
      std::cerr << "Error while loading ScanParquetFileTask: Should not happen " << std::endl;
      std::cerr << status.ToString() << std::endl;
   }
}
arrow::Status ScanParquetFileTask::init() {
   const size_t numWorkers = lingodb::scheduler::getNumWorkers();
   queryLifetimeChunks = new std::vector<std::deque<std::shared_ptr<ParquetBatchesWorkerResvState::ChunkWorkEntry>>>(numWorkers);
   //Keep in memory until query is finished
   //Joins for instance do not copy data and only refernces them and therefore the lifetime of the TableChunks has to exceed this table scan
   /* lingodb::runtime::getCurrentExecutionContext()->registerState({queryLifetimeChunks, [](void* ptr) {
                                                                     delete reinterpret_cast<std::vector<std::deque<LingoDBTable::TableChunk>>*>(ptr);
                                                                  }});*/
   std::shared_ptr<arrow::io::RandomAccessFile> input;
   ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(filePath));
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