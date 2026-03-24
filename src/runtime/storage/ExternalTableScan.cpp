#include "lingodb/runtime/storage/ExternalTableScan.h"

#include "arrow/record_batch.h"

#include <iostream>
#include <shared_mutex>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

namespace lingodb::runtime {

bool BatchesWorkerResvStateTest::hasMoreWork() {
   std::shared_lock<std::shared_mutex> stateLock(this->mutex);
   return resvCursor < unitAmount;
}

 std::pair<size_t, int> BatchesWorkerResvStateTest::fetchAndNextOwn(size_t splitSize, std::vector<std::deque<LingoDBTable::TableChunk>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, size_t numberOfRowGroups, std::unique_ptr<parquet::arrow::FileReader>& localReader) {
   std::unique_lock<std::shared_mutex> stateLock(this->mutex);
   auto workerId = lingodb::scheduler::currentWorkerId();
   size_t curr = resvCursor++;
   //Has more work localy on the "current" TableChunk!
   if (curr < unitAmount) {
      return {localChunkId, curr};
   }
   if (!rowGroupRecordBatchReader) {
      int newRgId = rgIdstartIndex.fetch_add(1);
      if (newRgId >= numberOfRowGroups) {
         return {localChunkId, -1};
      }
      std::vector<int> newRgIds = {newRgId};
      auto newlocalRowGroupReaderUncertain = localReader->GetRecordBatchReader(newRgIds);
      if (!newlocalRowGroupReaderUncertain.ok()) {
         //TODO handle error
         std::cerr << "Should not happen" << std::endl;
      }
      std::swap(rowGroupRecordBatchReader, newlocalRowGroupReaderUncertain.ValueOrDie());
      rgId = newRgId;
   }

   //If worker has no more localy work try to allocate new work in current rowGroup
   std::shared_ptr<arrow::RecordBatch> batch;
   auto status = rowGroupRecordBatchReader->ReadNext(&batch);
   if (!status.ok()) {
      std::cerr << "Error: Should not happen " << std::endl;
   }
   if (batch) {
      //new work found for current rowGroup
      auto& localChunks = (*queryLifetimeChunks)[workerId];
      //New Chunk
      LingoDBTable::TableChunk& chunk = localChunks.emplace_back(batch, 0);
      localChunkId = localChunks.size()-1;
      //TODO somehow give TableChunk back to caller
      unitAmount = (chunk.getNumRows() + splitSize - 1) / splitSize;
      resvCursor = 1;
      resvId = 0;
      //TODO assign batch id

      return {localChunkId, 0};
   } else {
      //No additional work found for current rowGroup, try to increase rowGroup to find new work
      int newRgId = rgIdstartIndex.fetch_add(1);
      if (newRgId >= numberOfRowGroups) {
         return {localChunkId, -1};
      }
      std::vector<int> newRgIds = {newRgId};
      auto newlocalRowGroupReaderUncertain = localReader->GetRecordBatchReader(newRgIds);
      if (!newlocalRowGroupReaderUncertain.ok()) {
         //TODO handle error
         std::cerr << "Should not happen " << std::endl;
      }
      std::swap(rowGroupRecordBatchReader, newlocalRowGroupReaderUncertain.ValueOrDie());
      rgId = newRgId;
      auto status2 = rowGroupRecordBatchReader->ReadNext(&batch);
      if (!status2.ok()) {
         std::cerr << "Error: Should not happen " << std::endl;
      }
      assert(batch);
      //Load new chunk from new rowgroup
      auto& localChunks = (*queryLifetimeChunks)[workerId];
      LingoDBTable::TableChunk& chunk = localChunks.emplace_back(batch, 0);
      localChunkId = localChunks.size()-1;
      //TODO somehow give TableChunk back to caller
      unitAmount = (chunk.getNumRows() + splitSize - 1) / splitSize;
      resvCursor = 1;
      resvId = 0;
      return {localChunkId, 0};
   }
};
std::pair<size_t, int> BatchesWorkerResvStateTest::fetchAndNext() {
   std::unique_lock<std::shared_mutex> resvLock(mutex);
   size_t cur = resvCursor++;
   return {localChunkId ,cur >= unitAmount ? -1 : cur};
}

//------------------------------------------------------
ScanParquetFileTask::ScanParquetFileTask(std::string filePath, std::vector<size_t> colids, std::function<void(BatchView*)> cb, std::unique_ptr<Restrictions> restrictions) : filePath(std::move(filePath)), colIds(std::move(colids)), cb(cb), restrictions(std::move(restrictions)) {
   init();
}
arrow::Status ScanParquetFileTask::init() {
   queryLifetimeChunks = new std::vector<std::deque<LingoDBTable::TableChunk>>(lingodb::scheduler::getNumWorkers());
   //Keep in memory until query is finished
   //Joins for instance do not copy data and only refernces them and therefore the lifetime of the TableChunks has to exceed this table scan
   lingodb::runtime::getCurrentExecutionContext()->registerState({queryLifetimeChunks, [](void* ptr) {
                                                                     delete reinterpret_cast<std::vector<std::deque<LingoDBTable::TableChunk>>*>(ptr);
                                                                  }});
   std::shared_ptr<arrow::io::RandomAccessFile> input;
   ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(filePath));

   for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
      localRowGroup.push_back(std::numeric_limits<int64_t>::max());
      arrayViewPtrs.push_back(std::vector<const ArrayView*>(colIds.size()));
      batchInfos.push_back(BatchView());
      batchInfos[i].arrays = arrayViewPtrs[i].data();
      selVecs.push_back(std::make_pair(new uint16_t[maxGeneratedScanMorselSize], new uint16_t[maxGeneratedScanMorselSize]));

      auto parquetFileReader = parquet::ParquetFileReader::Open(input);
      std::unique_ptr<parquet::arrow::FileReader> reader;

      numOfRowGroups = parquetFileReader->metadata()->num_row_groups();
      ARROW_RETURN_NOT_OK(parquet::arrow::FileReader::Make(
         arrow::default_memory_pool(), std::move(parquetFileReader), &reader));
      readers.emplace_back(std::move(reader));

      workerResvs.emplace_back(std::make_unique<BatchesWorkerResvStateTest>());
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
   auto [localGroupId, id] = state->fetchAndNextOwn(splitSize, queryLifetimeChunks, nextRowGroup, numOfRowGroups, readers[lingodb::scheduler::currentWorkerId()]);
   if (id != -1) {
      //Found new work
      state->resvId = id;
      state->localChunkId = localGroupId;
      return true;
   }

   //3. if the current worker has no more work locally and no more work globally, try to steal work from the worker we stole from last time
   if (state->stealWorkerId != std::numeric_limits<size_t>::max()) {
      auto* other = workerResvs[state->stealWorkerId].get();
      if (other->hasMoreWork()) {
         auto [otherChunkId, id] = other->fetchAndNext();
         if (id != -1) {
            state->resvId = id;
            state->localChunkId = otherChunkId;
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
            // only current worker can modify its onw stealWorkerId. no need to lock
            state->stealWorkerId = idx;
            state->resvId = id;
            state->localChunkId = otherChunkId;
            return true;
         }
      }
   }

   workExhausted.store(true);

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
   if (state->stealWorkerId != std::numeric_limits<size_t>::max()) {
      auto* other = workerResvs[state->stealWorkerId].get();
      auto& chunks = (*queryLifetimeChunks)[state->stealWorkerId];
      auto& chunk = chunks[state->localChunkId];
      unitRun(chunk);

   } else {
      auto& chunks = (*queryLifetimeChunks)[workerId];
      //Last chunk is our current working chunk right?
      auto& chunk = chunks[state->localChunkId];
      unitRun(chunk);
   }

   //Load batches
  /* std::shared_ptr<arrow::RecordBatch> batch;
   while (reader->ReadNext(&batch).ok() && batch) {
      //Add batch to list, to keep it
      LingoDBTable::TableChunk& chunk = chunks.emplace_back(batch, 0);
      for (size_t start = 0; start < static_cast<size_t>(batch->num_rows()); start += maxGeneratedScanMorselSize) {
         size_t len = std::min(maxGeneratedScanMorselSize, static_cast<size_t>(batch->num_rows()) - start);
         batchView.offset = start;
         batchView.length = len;
         for (size_t i = 0; i < colIds.size(); i++) {
            batchView.arrays[i] = chunk.getArrayView(colIds[i]);
         }
         auto [newLen, selVec] = restrictions->applyFilters(start, len, selVec1, selVec2, [&](size_t colId) { return chunk.getArrayView(colId); });
         batchView.length = newLen;
         batchView.selectionVector = selVec;
         if (batchView.length > 0) {
            assert(cb);
            cb(&batchView);
         }
      }
   }*/
}
ScanParquetFileTask::~ScanParquetFileTask() {
   for (auto& [selVec1, selVec2] : selVecs) {
      delete[] selVec1;
      delete[] selVec2;
   }
}

}