#ifndef LINGODB_RUNTIME_STORAGE_EXTERNALTABLESCAN_H
#define LINGODB_RUNTIME_STORAGE_EXTERNALTABLESCAN_H

#include <atomic>
#include <cstdint>
#include <limits>

#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/runtime/storage/Restrictions.h"
#include "lingodb/scheduler/Tasks.h"

#include <arrow/dataset/file_parquet.h>

#include <shared_mutex>

namespace lingodb::runtime {
struct alignas(64) ParquetBatchesWorkerResvState {
   public:
   ///Describes the Work
   class WorkInfo {
      public:
      size_t ownChunkId; //The current chunk the work is alloacated to
      long currentMorsel; //The current morsel inside the chunk
      bool buffering = false;
   };

   std::shared_mutex mutex;

   /// Current rowGroup
   size_t rgId{0};
   /// Chunk id currently owned by this worker/state
   size_t ownChunkId{0};
   // Chunk id reserved for the current morsel on this worker (may not refer to own or stolen chunk).
   // Only accessed by the owning worker.
   size_t reservedChunkId{0};

   long resvCursor{0};
   size_t resvId{0};
   /// worker id steal tasks from
   size_t stealWorkerId{std::numeric_limits<size_t>::max()};
   /// resvId gotten from @stealWorkerId
   size_t stolenResvId{0};
   /// chunkId gotten from @stealWorkerId
   size_t stolenResvChunkId{0};

   long unitAmount{0};

   std::unique_ptr<arrow::RecordBatchReader> rowGroupRecordBatchReader;
   arrow::Future<> buffering = arrow::Future<>::MakeFinished();

   // Set to true once this worker/state can never produce more work.
   // Used for global termination detection in the scan task.
   std::atomic<bool> fullyExhausted{false};

   bool hasMoreWork();
   /**
    *
    * @param splitSize number of rows per morsel
    * @param queryLifetimeChunks
    * @param rgIdstartIndex atomic next rowgroup counter
    * @param numberOfRowGroups overall number of rowgroups
    * @param colIds
    * @param localReader reader of the current thread
    * @return pair of local chunk id and reservation id. reservation id is -1 if no more work could be found, otherwise it is the id of the reservation
    */
   WorkInfo fetchAndNextOwn(size_t splitSize, std::vector<std::deque<LingoDBTable::TableChunk>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader);
   std::pair<size_t, int> fetchAndNext();

   private:
   void initNewRowGroup(int rowGroup, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader);
};
class ScanParquetFileTask : public scheduler::TaskWithImplicitContext {
   std::string filePath;
   std::function<void(BatchView*)> cb;
   std::unique_ptr<Restrictions> restrictions;
   size_t numOfRowGroups;

   std::vector<std::unique_ptr<parquet::arrow::FileReader>> readers;

   std::vector<int> colIds;

   std::atomic<int> nextRowGroup = 0;

   std::vector<size_t> localRowGroup;
   std::vector<BatchView> batchInfos;
   std::vector<std::vector<const ArrayView*>> arrayViewPtrs;

   std::vector<std::deque<LingoDBTable::TableChunk>>* queryLifetimeChunks = nullptr;
   std::vector<std::pair<uint16_t*, uint16_t*>> selVecs;

   std::vector<std::unique_ptr<ParquetBatchesWorkerResvState>> workerResvs;
   size_t splitSize{20000};

   public:
   ScanParquetFileTask(std::string filePath, std::vector<int> colids, std::function<void(BatchView*)> cb, std::unique_ptr<Restrictions> restrictions);
   arrow::Status init();

   void unitRun(LingoDBTable::TableChunk& chunk);

   bool allocateWork() override;
   void performWork() override;
   ~ScanParquetFileTask();
};
} // namespace lingodb::runtime
#endif
