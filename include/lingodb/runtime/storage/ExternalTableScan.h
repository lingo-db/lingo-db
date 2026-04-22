#ifndef LINGODB_RUNTIME_STORAGE_EXTERNALTABLESCAN_H
#define LINGODB_RUNTIME_STORAGE_EXTERNALTABLESCAN_H

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/runtime/storage/Restrictions.h"
#include "lingodb/scheduler/Tasks.h"

#include <shared_mutex>
#include <arrow/dataset/file_parquet.h>
#include <parquet/statistics.h>
#ifndef KEEP_IN_MEMEORY
#define KEEP_IN_MEMEORY 1
#endif

// Backward-compatible alias.
#ifndef KEEP_IN_MEMORY
#define KEEP_IN_MEMORY KEEP_IN_MEMEORY
#endif
namespace lingodb::runtime {
struct MetadataFilter {
   virtual ~MetadataFilter() = default;
   virtual bool apply(std::shared_ptr<parquet::Statistics> stats) const = 0;
};
template <class T>
struct MinMaxMetadataFilter : public MetadataFilter {
   public:
   MinMaxMetadataFilter(std::optional<T> min, std::optional<T> max, bool minInclusive = false, bool maxInclusive = false) : min(std::move(min)), max(std::move(max)), minInclusive(minInclusive), maxInclusive(maxInclusive) {
   }
   std::optional<T> min;
   std::optional<T> max;
   bool minInclusive = false;
   bool maxInclusive = false;
   bool apply(std::shared_ptr<parquet::Statistics> stats) const override;
};

struct NotNullFilter : public MetadataFilter {
   public:
   NotNullFilter() {
   }
   /**
    *
    * @param stats
    * @return true if there are no null values inside - false if there are only null values
    */
   bool apply(std::shared_ptr<parquet::Statistics> stats) const override;
};
struct NullFilter : public MetadataFilter {
   public:
   NullFilter() {
   }
   /**
    * @param stats
    * @return true if there are null values inside - false if there are no null values
    */
   bool apply(std::shared_ptr<parquet::Statistics> stats) const override;
};
class ParquetBatchesWorkerResvState {
   public:
   struct ChunkWorkEntry {
      std::shared_ptr<LingoDBTable::TableChunk> chunk;
      long totalMorsels{0};
      std::atomic<long> completedMorsels{0};
   };

   class WorkInfo {
      public:
      size_t ownChunkId; //The current chunk the work is alloacated to
      long currentMorsel; //The current morsel inside the chunk
      bool buffering = false;
   };
   size_t workerId;
   size_t rgId{0};
   std::vector<std::pair<int, std::shared_ptr<MetadataFilter>>> metadataFilters;
   // Chunk id currently owned by this worker/state. Other workers may read this when stealing.
   size_t ownChunkId{0};
   // Chunk id reserved for the current morsel on this worker (may refer to own or stolen chunk).
   // Only accessed by the owning worker.
   size_t reservedChunkId{0};
   std::shared_mutex mutex;
   long resvCursor{0};
   size_t resvId{0};
   // worker id steal tasks from
   size_t stealWorkerId{std::numeric_limits<size_t>::max()};

   // Set to true once this worker/state can never produce more work.
   // Used for global termination detection in the scan task.
   std::atomic<bool> fullyExhausted{false};

   long unitAmount{0};

   std::unique_ptr<arrow::RecordBatchReader> rowGroupRecordBatchReader;
   std::atomic<bool> isBuffering = false;
   int prefetchedRgId = -1;

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
   WorkInfo fetchAndNextOwn(size_t splitSize, std::vector<std::deque<std::shared_ptr<ChunkWorkEntry>>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader);

   std::pair<size_t, int> fetchAndNext();

   private:
   bool rowGroupPassesMetadataFilters(int rowGroup, std::unique_ptr<parquet::arrow::FileReader>& localReader) const;
   int fetchNextMatchingRowGroup(std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::unique_ptr<parquet::arrow::FileReader>& localReader) const;
   void initNewRowGroup(int rowGroup, size_t splitSize, std::vector<std::deque<std::shared_ptr<ChunkWorkEntry>>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader);
   /**
    *
    * @param splitSize
    * @param queryLifetimeChunks
    * @param rgIdstartIndex
    * @param numberOfRowGroups
    * @param colIds
    * @param localReader
    * @return
    * This method loads the next record batch. It should only be called when no buffering is ongoing. If the current row group is exhausted, it initializes and prefetches the next one.
    */
   WorkInfo tryFetchNextRecordBatch(size_t splitSize, std::vector<std::deque<std::shared_ptr<ChunkWorkEntry>>>* queryLifetimeChunks, std::atomic<int>& rgIdstartIndex, int numberOfRowGroups, std::vector<int>& colIds, std::unique_ptr<parquet::arrow::FileReader>& localReader);
};
class ScanParquetFileTask : public scheduler::TaskWithImplicitContext {
   std::string filePath;
   std::function<void(BatchView*)> cb;
   std::unique_ptr<Restrictions> restrictions;
   /**
    * .first columnId
    * .second filter definition
    */
   std::vector<std::pair<int, std::shared_ptr<MetadataFilter>>> metadataFilters;
   size_t numOfRowGroups;

   std::vector<std::unique_ptr<parquet::arrow::FileReader>> readers;

   std::vector<int> colIds;

   std::atomic<int> nextRowGroup = 0;

   std::vector<size_t> localRowGroup;
   std::vector<BatchView> batchInfos;
   std::vector<std::vector<const ArrayView*>> arrayViewPtrs;

   std::vector<std::deque<std::shared_ptr<ParquetBatchesWorkerResvState::ChunkWorkEntry>>>* queryLifetimeChunks = nullptr;
   std::vector<std::pair<uint16_t*, uint16_t*>> selVecs;

   std::vector<std::unique_ptr<ParquetBatchesWorkerResvState>> workerResvs;
   size_t splitSize{20000};

   public:
   ScanParquetFileTask(std::string filePath, std::vector<int> colids, std::function<void(BatchView*)> cb, std::unique_ptr<Restrictions> restrictions, std::vector<FilterDescription> filterDescriptions);
   arrow::Status init(std::vector<FilterDescription>& filterDescriptions);

   void unitRun(LingoDBTable::TableChunk& chunk);

   bool allocateWork() override;
   void performWork() override;
   ~ScanParquetFileTask();
};
} // namespace lingodb::runtime
#endif
