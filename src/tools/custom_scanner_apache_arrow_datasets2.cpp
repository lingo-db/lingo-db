
#define MULTITHREADED 1
#include <iostream>
#include <string>

#include "arrow/compute/expression.h"
#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/scheduler/Tasks.h"

#include <thread>
#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/cast.h>
#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/discovery.h>
#include <arrow/dataset/file_base.h>
#include <arrow/dataset/file_ipc.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/dataset/plan.h>
#include <arrow/dataset/scanner.h>
#include <arrow/filesystem/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/util/iterator.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

class ScanParquetFileTask : public lingodb::scheduler::Task {
   std::string filePath;
   size_t numOfRowGroups;
   std::shared_ptr<parquet::FileMetaData> metadata;

   std::unique_ptr<parquet::arrow::FileReader> reader;
   std::mutex readerMutex;

   std::vector<size_t> colIds;

   std::atomic<int> nextRowGroup = 0;

   std::vector<size_t> localRowGroup;
   std::vector<lingodb::runtime::BatchView> batchInfos;
   std::vector<std::vector<const lingodb::runtime::ArrayView*>> arrayViewPtrs;

   public:
   ScanParquetFileTask(std::string filePath, std::vector<size_t> colIds) : filePath(std::move(filePath)), colIds(colIds) {
      init();
   }
   arrow::Status init() {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         localRowGroup.push_back(std::numeric_limits<int>::max());
         batchInfos.emplace_back(lingodb::runtime::BatchView());
         arrayViewPtrs.emplace_back(std::vector<const lingodb::runtime::ArrayView*>(colIds.size()));
         batchInfos[i].arrays = arrayViewPtrs[i].data();
      }

      std::shared_ptr<arrow::io::RandomAccessFile> input;
      ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(filePath));
      auto parquetFileReader = parquet::ParquetFileReader::Open(input);
      metadata = parquetFileReader->metadata();
      numOfRowGroups = metadata->num_row_groups();

      ARROW_RETURN_NOT_OK(parquet::arrow::FileReader::Make(
         arrow::default_memory_pool(), std::move(parquetFileReader), &reader));

      return arrow::Status::OK();
   }

   bool allocateWork() override {
      int rgId = nextRowGroup.fetch_add(1);

      if (rgId >= numOfRowGroups) {
         workExhausted = true;
         return false;
      }
      localRowGroup[lingodb::scheduler::currentWorkerId()] = rgId;

      return true;
   }
   void performWork() override {
      int rgId = localRowGroup[lingodb::scheduler::currentWorkerId()];
      std::vector<int> row_groups{rgId};
      if (workExhausted) {
         return;
      }
      readerMutex.lock();
      auto rowGroupBatchReaderUncertain = reader->GetRecordBatchReader(row_groups);
      readerMutex.unlock();
      if (!rowGroupBatchReaderUncertain.ok()) {
         //TODO handle err
         std::cerr << "Error getting batch reader for row group " << rgId << ": " << rowGroupBatchReaderUncertain.status().ToString() << "\n";
      }

      std::unique_ptr<arrow::RecordBatchReader>* rowGroupBatchReader = &rowGroupBatchReaderUncertain.ValueOrDie();
      while (true) {
         auto batchUncertain = (*rowGroupBatchReader)->Next();
         if (!batchUncertain.ok()) {
            //TODO handle error
         }
         auto batch = batchUncertain.ValueOrDie();
         if (!batch) {
            return; // No more batches
         }
         lingodb::runtime::LingoDBTable::TableChunk chunk{batch, 0};

         lingodb::runtime::BatchView& batchView = batchInfos[lingodb::scheduler::currentWorkerId()];
         batchView.offset = 0;
         batchView.selectionVector = lingodb::runtime::BatchView::defaultSelectionVector.data();
         batchView.length = batch->num_rows();

         for (size_t i = 0; i < colIds.size(); i++) {
            batchView.arrays[i] = chunk.getArrayView(colIds[i]);
         }

         std::cerr << "Processing batch with " << batch->num_rows() << " rows from row group " << rgId << " tId" << std::this_thread::get_id() << "\n";
      }
   }
};

arrow::Status Test() {
   auto scheduler = lingodb::scheduler::startScheduler(1);
   std::string inputFilePath = "/home/bachmaier/projects/lingo-db/resources/data/tpch-1/nation.parquet";
   std::vector<size_t> colIds{0};
   auto scanTaskUnique = std::make_unique<ScanParquetFileTask>(inputFilePath, colIds);
   lingodb::scheduler::awaitEntryTask(std::move(scanTaskUnique));
   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
   return arrow::Status::OK();
}

int main(int argc, char** argv) {
   auto status = Test();
   if (!status.ok()) {
      std::cerr << "Error: " << status.message() << std::endl;
      return 1;
   }
   return 0;
}
