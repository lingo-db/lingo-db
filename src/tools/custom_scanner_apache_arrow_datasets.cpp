
#define MULTITHREADED 1
#include <iostream>
#include <string>

#include "arrow/compute/expression.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Task.h"

#include <condition_variable>
#include <mutex>
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
#include <arrow/ipc/writer.h>
#include <arrow/util/iterator.h>
#include <parquet/arrow/writer.h>

#include <thread>
#include <arrow/csv/reader.h>
#include <arrow/util/async_generator.h>
#include <bits/this_thread_sleep.h>

class ArrowScanTask : public lingodb::scheduler::Task {
   class alignas(64) ArrowSubScanTaskInfo {
      public:
      arrow::internal::FnOnce<void()> task;
      arrow::StopToken stop_token;
      arrow::internal::Executor::StopCallback stop_callback;
      arrow::internal::TaskHints hints;
      void operator()(std::function<void(ArrowSubScanTaskInfo*)> releaseTaskInfo) {
         if (!task)
            return;
         if (!stop_token.IsStopRequested()) {
            std::move(task)();
         } else if (stop_callback) {
            std::move(stop_callback)(stop_token.Poll());
         }
         releaseTaskInfo(this);
      }
   };

   public:
   ArrowScanTask() {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         currentTasks_.push_back(nullptr);
      }
   }

   void releaseTaskInfo(ArrowSubScanTaskInfo* t) {
   }

   void add_arrow_task(ArrowSubScanTaskInfo task_info) {
   }

   arrow::Status addTask(arrow::internal::TaskHints hints, arrow::internal::FnOnce<void()> task, arrow::StopToken stopToken,
                         arrow::internal::Executor::StopCallback&& stop_callback) {
      if (workExhausted) {
         assert(false);
      }

      auto task_info = std::make_unique<ArrowSubScanTaskInfo>();
      task_info->task = std::move(task);
      task_info->stop_token = stopToken;
      task_info->stop_callback = std::move(stop_callback);
      task_info->hints = hints;
      {
         static size_t x = 0;
         std::unique_lock<std::mutex> lock(mutex);

         x++;
         std::cerr << x << std::endl;
         taskQueue_.push(std::move(task_info));
      }
      hasWorkInQueueCondVariable.notify_all();
      return arrow::Status::OK();
   }

   arrow::Result<std::shared_ptr<arrow::csv::StreamingReader>> result() {
   }

   bool allocateWork() override {
      std::unique_lock<std::mutex> lock(mutex);

      if (taskQueue_.empty()) {
         hasWorkInQueueCondVariable.wait(lock, [this]() { return !taskQueue_.empty() || workExhausted; });
      }
      if (workExhausted) {
         return false;
      }

      auto& top = taskQueue_.front();

      if (!top) {
         throw std::runtime_error("SHOULD NOT HAPPEN");
      }
      currentTasks_[lingodb::scheduler::currentWorkerId()] = std::move(top);
      taskQueue_.pop();

      return true;
   }
   void performWork() override {
      auto& currentTask = currentTasks_[lingodb::scheduler::currentWorkerId()];
      if (!currentTask) return;
      (*currentTask)([this](ArrowSubScanTaskInfo* t) { this->releaseTaskInfo(t); });
      currentTasks_[lingodb::scheduler::currentWorkerId()] = nullptr;
   }
   void setup() override {
   }
   void teardown() override {
   }
   void finish() {
      std::unique_lock<std::mutex> lock(mutex);
      workExhausted.store(true);
      std::cerr << "Finish \n";
      hasWorkInQueueCondVariable.notify_all();
   }
   ~ArrowScanTask() override {
   }

   private:
   static constexpr size_t POOL_SIZE = 128;
   std::queue<std::unique_ptr<ArrowSubScanTaskInfo>> taskQueue_;
   std::vector<std::unique_ptr<ArrowSubScanTaskInfo>> currentTasks_;

   std::condition_variable hasWorkInQueueCondVariable;

   std::mutex mutex;
};

class ArrowScanTaskWrapper : public arrow::internal::Executor {
   public:
   ArrowScanTaskWrapper(ArrowScanTask* scanTask) : scanTask(scanTask) {
   }

   arrow::Status SpawnReal(arrow::internal::TaskHints hints, arrow::internal::FnOnce<void()> task, arrow::StopToken stopToken,
                           StopCallback&& stop_callback) override {
      return scanTask->addTask(hints, std::move(task), stopToken, std::move(stop_callback));
   }

   int GetCapacity() override {
      return lingodb::scheduler::getNumWorkers();
   }

   ~ArrowScanTaskWrapper() override {
      std::cerr << "Delete" << std::endl;
      scanTask->finish();
   }

   private:
   ArrowScanTask* scanTask;
};
static int count = 0;
arrow::Status ProcessBatch(const arrow::dataset::TaggedRecordBatch& tagged_batch) {
   std::cerr << "ThreadId " << std::this_thread::get_id() << " got batch with " << tagged_batch.record_batch->num_rows() << " rows\n";
   std::this_thread::sleep_for(std::chrono::milliseconds(2000));
   count += tagged_batch.record_batch->num_rows();
   return arrow::Status::OK();
}

arrow::Status ScanWholeDataset(
   const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
   const std::shared_ptr<arrow::dataset::FileFormat>& format, const std::string& base_dir, ArrowScanTask* scanTask, bool use_thread) {
   auto customOption = std::make_shared<arrow::dataset::ScanOptions>();
   auto scanWrapper = std::make_shared<ArrowScanTaskWrapper>(scanTask);
   customOption->cpu_executor = scanWrapper.get();
   customOption->io_context = arrow::io::IOContext(arrow::default_memory_pool(), scanWrapper.get());
   customOption->batch_size = 1024 * 1024;
   customOption->use_threads = true;

   auto parquet_options = std::make_shared<arrow::dataset::ParquetFragmentScanOptions>();
   parquet_options->arrow_reader_properties->set_use_threads(false);
   // 1. Disable Scanner-level readahead (prevents Acero from queueing background fragment fetching)
   customOption->fragment_readahead = 0;

   // 2. Disable Parquet Pre-buffering (stops the global I/O thread pool from fetching byte ranges in the background)
   parquet_options->arrow_reader_properties->set_pre_buffer(false);

   // 3. Disable the Dict pushdown if it's triggering async dictionary decoding
   parquet_options->arrow_reader_properties->set_read_dictionary(0, false); // Optional, but helps isolate

   customOption->fragment_scan_options = parquet_options;
   customOption->fragment_scan_options = parquet_options;

   arrow::fs::FileSelector selector;
   selector.base_dir = base_dir;
   selector.recursive = true;

   auto start = std::chrono::high_resolution_clock::now();

   ARROW_ASSIGN_OR_RAISE(
      auto factory, arrow::dataset::FileSystemDatasetFactory::Make(filesystem, selector, format, arrow::dataset::FileSystemFactoryOptions()));

   ARROW_ASSIGN_OR_RAISE(auto dataset, factory->Finish());
   // Read the entire dataset as a Table
   arrow::dataset::ScannerBuilder scan_builder(dataset, customOption);
   assert(scan_builder.GetScanOptions().ValueOrDie()->cpu_executor == scanWrapper.get());

   ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder.Finish());
   assert(scanner->options()->cpu_executor == scanWrapper.get());
   scanner->Scan([&](auto x) { return arrow::Status::OK(); });
   ARROW_RETURN_NOT_OK(scanner->Scan([&](const arrow::dataset::TaggedRecordBatch taggedBatch) {
      ARROW_RETURN_NOT_OK(scanWrapper->Spawn([batch = std::move(taggedBatch)]() {
         ProcessBatch(batch).ok(); // handle error properly
      }));
      return arrow::Status::OK();
   }));

   auto end = std::chrono::high_resolution_clock::now();
   std::cerr << "Result  " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms\n";

   return arrow::Status::OK();
}

arrow::Status Test() {
   auto scheduler = lingodb::scheduler::startScheduler(2);
   auto scanTaskUnique = std::make_unique<ArrowScanTask>();
   auto* scanTaskPtr = scanTaskUnique.get();
   lingodb::scheduler::enqueueTask(std::move(scanTaskUnique));
   ARROW_RETURN_NOT_OK(arrow::compute::Initialize());

   std::string base_path = "/home/bachmaier/projects/lingo-db/_tmp/lineitem_parquet/";
   std::string root_path;
   std::string uri = "file:///home/bachmaier/projects/lingo-db/_tmp";
   ARROW_ASSIGN_OR_RAISE(auto fs, arrow::fs::FileSystemFromUri(uri, &root_path));
   auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();

   ARROW_RETURN_NOT_OK(ScanWholeDataset(fs, format, base_path, scanTaskPtr, true));

   return arrow::Status::OK();
}

class Scan {
   public:
   Scan(std::string uri, std::string folderDir) : uri(std::move(uri)), folderDir(std::move(folderDir)) {
   }

   private:
   std::string uri;
   std::string folderDir;
   //TODO other configs
};

int main(int argc, char** argv) {
   auto status = Test();
   if (!status.ok()) {
      std::cerr << "Error: " << status.message() << std::endl;
      return 1;
   }
   return 0;
}
