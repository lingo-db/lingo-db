#pragma once
#include <atomic>
#include <cstdint>
#include <limits>

#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/runtime/storage/Restrictions.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"

#include <shared_mutex>
#include <arrow/dataset/file_parquet.h>
static constexpr size_t maxGeneratedScanMorselSize = std::numeric_limits<int16_t>().max();

namespace lingodb::runtime {
class ParquetBatchesWorkerResvState {
   public:
   class WorkInfo {
      size_t localChunkId; //The current chunk the work is alloacated to
      size_t curr; //The current morsel inside the chunk
      arrow::Future<> ioFinished; //Future to indicate whether the prefetching is finished

   };
   size_t rgId{0};
   // Chunk id currently owned by this worker/state. Other workers may read this when stealing.
   size_t ownChunkId{0};
   // Chunk id reserved for the current morsel on this worker (may refer to own or stolen chunk).
   // Only accessed by the owning worker.
   size_t reservedChunkId{0};
   std::shared_mutex mutex;
   size_t resvCursor{0};
   size_t resvId{0};
   // worker id steal tasks from
   size_t stealWorkerId{std::numeric_limits<size_t>::max()};

   // Set to true once this worker/state can never produce more work.
   // Used for global termination detection in the scan task.
   std::atomic<bool> fullyExhausted{false};

   size_t unitAmount{0};

   std::unique_ptr<arrow::RecordBatchReader> rowGroupRecordBatchReader;

   bool hasMoreWork();
   /**
    *
    * @param splitSize number of rows per morsel
    * @param queryLifetimeChunks
    * @param rgIdstartIndex atomic next rowgroup counter
    * @param numberOfRowGroups overall number of rowgroups
    * @param localReader reader of the current thread
    * @return pair of local chunk id and reservation id. reservation id is -1 if no more work could be found, otherwise it is the id of the reservation
    */
   std::pair<size_t, int> fetchAndNextOwn(size_t splitSize, std::vector<std::deque<LingoDBTable::TableChunk>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, size_t numberOfRowGroups, std::unique_ptr<parquet::arrow::FileReader>& localReader);
   std::pair<size_t, int> fetchAndNext();
};
class ScanParquetFileTask : public scheduler::TaskWithImplicitContext {
   std::string filePath;
   std::function<void(BatchView*)> cb;
   std::unique_ptr<Restrictions> restrictions;
   size_t numOfRowGroups;

   std::vector<std::unique_ptr<parquet::arrow::FileReader>> readers;

   std::vector<size_t> colIds;

   std::atomic<int> nextRowGroup = 0;

   std::vector<size_t> localRowGroup;
   std::vector<BatchView> batchInfos;
   std::vector<std::vector<const ArrayView*>> arrayViewPtrs;

   std::vector<std::deque<LingoDBTable::TableChunk>>* queryLifetimeChunks = nullptr;
   std::vector<std::pair<uint16_t*, uint16_t*>> selVecs;

   std::vector<std::unique_ptr<ParquetBatchesWorkerResvState>> workerResvs;
   size_t splitSize{20000};

   public:
   ScanParquetFileTask(std::string filePath, std::vector<size_t> colids, std::function<void(BatchView*)> cb, std::unique_ptr<Restrictions> restrictions);
   arrow::Status init();

   void unitRun(LingoDBTable::TableChunk& chunk);

   bool allocateWork() override;
   void performWork() override;
   ~ScanParquetFileTask();
};
}