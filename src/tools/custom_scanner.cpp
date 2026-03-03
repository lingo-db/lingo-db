#include "features.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/dataset/api.h>
#include <arrow/io/api.h>

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <string>
#include <arrow/util/formatting.h>
#define MULTITHREADED 1
namespace {
lingodb::utility::GlobalSetting<bool> eagerLoading("system.eager_loading", false);
}

class ArrowScanTask : public lingodb::scheduler::Task, public arrow::internal::Executor {
   class alignas(64) ArrowSubScanTaskInfo {
      public:
      arrow::internal::FnOnce<void()> task;
      arrow::StopToken stop_token;
      arrow::internal::Executor::StopCallback stop_callback;
      arrow::internal::TaskHints hints;
      std::shared_ptr<ArrowSubScanTaskInfo> next;
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
   ArrowScanTask(std::unique_ptr<lingodb::scheduler::SchedulerHandle> scheduler, std::shared_ptr<arrow::io::ReadableFile> inputFile, arrow::csv::ReadOptions csvReadOption, arrow::csv::ParseOptions csvParseOptions, arrow::csv::ConvertOptions csvConvertOptions) : scheduler_(std::move(scheduler)) {
      csvReadOption.use_threads = true;
      currentTasks_.resize(lingodb::scheduler::getNumWorkers(), nullptr);
      localRowCounts_.resize(lingodb::scheduler::getNumWorkers(), 0);

      // Pre-populate the free pool for better performance
      for (size_t i = 0; i < POOL_SIZE / 2; i++) {
         freePool_.push(new ArrowSubScanTaskInfo());
      }

      ioContext_ = arrow::io::IOContext(arrow::default_memory_pool(), this);

      reader_ = arrow::csv::StreamingReader::MakeAsync(ioContext_,
                                                       inputFile,
                                                       this,
                                                       csvReadOption,
                                                       csvParseOptions,
                                                       csvConvertOptions);
   }
   ArrowSubScanTaskInfo* allocateTaskInfo() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (freePool_.empty()) {
         return new ArrowSubScanTaskInfo();
      }
      auto t = freePool_.front();
      freePool_.pop();
      return t;
   }

   void releaseTaskInfo(ArrowSubScanTaskInfo* t) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (freePool_.size() < POOL_SIZE) {
         freePool_.push(t);
      } else {
         delete t;
      }
   }

   void add_arrow_task(ArrowSubScanTaskInfo* task_info) {
      std::unique_lock<std::mutex> lock(mutex_);
      taskQueue_.push(task_info);
      hasWorkInQueueCondVariable.notify_one();
   }

   arrow::Status SpawnReal(arrow::internal::TaskHints hints, arrow::internal::FnOnce<void()> task, arrow::StopToken stopToken,
                           StopCallback&& stop_callback) override {
      auto task_info = allocateTaskInfo();
      task_info->task = std::move(task);
      task_info->stop_token = stopToken;
      task_info->stop_callback = std::move(stop_callback);
      task_info->hints = hints;
      add_arrow_task(task_info);
      return arrow::Status::OK();
   }

   int GetCapacity() override {
      return lingodb::scheduler::getNumWorkers();
   }

   arrow::Result<std::shared_ptr<arrow::csv::StreamingReader>> result() {
      auto r = reader_.result();
      return r;
   }

   void addBatch(std::shared_ptr<arrow::RecordBatch> batch) {
      std::unique_lock<std::mutex> lock(batchMutex_);
      recordBatches_.push_back(batch);
   }

   void processRowsMultithreaded() {
   }

   std::vector<std::shared_ptr<arrow::RecordBatch>>& getRecordBatches() {
      return recordBatches_;
   }

   size_t getTotalRowCount() const {
      return totalRowCount_;
   }

   bool allocateWork() override {
      std::unique_lock<std::mutex> lock(mutex_);

      if (taskQueue_.empty()) {
         hasWorkInQueueCondVariable.wait(lock, [this]() { return !taskQueue_.empty(); });
      }
      auto top = taskQueue_.front();

      if (!top) {
         return false;
      }
      currentTasks_[lingodb::scheduler::currentWorkerId()] = top;
      taskQueue_.pop();

      return true;
   }
   void performWork() override {
      auto currentTask = currentTasks_[lingodb::scheduler::currentWorkerId()];
      if (!currentTask) return;
      (*currentTask)([this](ArrowSubScanTaskInfo* t) { this->releaseTaskInfo(t); });
   }
   void setup() override {
   }
   void teardown() override {
   }
   void finish() {
      workExhausted.store(true);
   }
   ~ArrowScanTask() override {
      while (!freePool_.empty()) {
         auto* ptr = freePool_.front();
         freePool_.pop();
         delete ptr;
      }
   }

   private:
   static constexpr size_t POOL_SIZE = 128;
   std::queue<ArrowSubScanTaskInfo*> taskQueue_;
   std::queue<ArrowSubScanTaskInfo*> freePool_;
   std::unique_ptr<lingodb::scheduler::SchedulerHandle> scheduler_;
   std::vector<ArrowSubScanTaskInfo*> currentTasks_;
   arrow::Future<std::shared_ptr<arrow::csv::StreamingReader>> reader_;
   arrow::io::IOContext ioContext_;
   alignas(64) std::mutex mutex_;
   std::condition_variable hasWorkInQueueCondVariable;

   // Row counting and batch storage
   std::mutex batchMutex_;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches_;
   std::vector<size_t> localRowCounts_;
   std::atomic<size_t> totalRowCount_{0};
};

class CSVScanner {
   public:
   CSVScanner(const std::string& file_path) : file_path_(file_path) {}

   arrow::Result<size_t> Init() {
      auto scheduler = lingodb::scheduler::startScheduler();
      // Open the CSV file
      ARROW_ASSIGN_OR_RAISE(auto input_file,
                            arrow::io::ReadableFile::Open(file_path_));

      auto csv_read_opts = arrow::csv::ReadOptions::Defaults();
      csv_read_opts.use_threads = MULTITHREADED;

      auto csv_parse_opts = arrow::csv::ParseOptions::Defaults();
      auto csv_convert_opts = arrow::csv::ConvertOptions::Defaults();
      auto init_start = std::chrono::high_resolution_clock::now();
      ArrowScanTask* scanTaskPtr = nullptr;

#if MULTITHREADED

      auto scanTask = std::make_unique<ArrowScanTask>(std::move(scheduler), input_file, csv_read_opts, csv_parse_opts, csv_convert_opts);
      scanTaskPtr = scanTask.get();
      lingodb::scheduler::enqueueTask(std::move(scanTask));
      ARROW_ASSIGN_OR_RAISE(reader_, scanTaskPtr->result());

#else
      auto io_context = arrow::io::IOContext(arrow::default_memory_pool());
      ARROW_ASSIGN_OR_RAISE(reader_, arrow::csv::StreamingReader::Make(io_context, input_file, csv_read_opts, csv_parse_opts, csv_convert_opts));
#endif

      auto reader_ptr = std::dynamic_pointer_cast<arrow::csv::StreamingReader>(reader_);
      if (!reader_ptr) {
         return arrow::Status::Invalid("Failed to cast reader");
      }

      {
         while (true) {
            ARROW_ASSIGN_OR_RAISE(auto batch, reader_ptr->Next());
            if (!batch) break;
            record_batches_.push_back(batch);
         }
      }
      if (scanTaskPtr) {
         scanTaskPtr->finish();
      }

      // Mark that no more work will be scheduled from the Arrow reader

      auto init_end = std::chrono::high_resolution_clock::now();
      auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
      std::cout << "CSV Initialization Total: " << init_ms << "ms" << std::endl;

      return init_ms;
   }

   arrow::Status Scan() {
      auto scan_start = std::chrono::high_resolution_clock::now();

      if (record_batches_.empty()) {
         return arrow::Status::Invalid("No data loaded. Call Init() first.");
      }

      std::cout << "\nCSV Scanner Results:" << std::endl;
      std::cout << "-------------------" << std::endl;

      size_t total_rows = 0;
      for (size_t i = 0; i < record_batches_.size(); ++i) {
         auto& batch = record_batches_[i];

         // Print schema from first batch
         if (i == 0) {
            std::cout << "\nSchema:" << std::endl;
            for (int j = 0; j < batch->num_columns(); ++j) {
               std::cout << "  - " << batch->column(j)->type()->ToString()
                         << " " << batch->schema()->field(j)->name()
                         << std::endl;
            }
            std::cout << std::endl;
         }

         total_rows += batch->num_rows();
      }

      std::cout << "Total rows scanned: " << total_rows << std::endl;

      auto scan_end = std::chrono::high_resolution_clock::now();
      auto scan_ms = std::chrono::duration_cast<std::chrono::milliseconds>(scan_end - scan_start).count();
      std::cout << "CSV Scanning Total: " << scan_ms << "ms" << std::endl;
      return arrow::Status::OK();
   }

   std::vector<std::shared_ptr<arrow::RecordBatch>>& GetBatches() {
      return record_batches_;
   }

   private:
   std::string file_path_;
   std::shared_ptr<arrow::RecordBatchReader> reader_;
   std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
};

int main(int argc, char** argv) {
   using namespace lingodb;

   if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <csv_file>" << std::endl;
      return 1;
   }

   std::string csv_file = argv[1];
   CSVScanner scanner(csv_file);

   // Initialize the scanner
   for (size_t i = 1; i < 2; ++i) {
      auto init_status = scanner.Init();
      if (!init_status.ok()) {
         std::cerr << "Error initializing scanner: " << init_status.status().message()
                   << std::endl;
         return 1;
      }
   }

   // Scan the CSV file
   auto scan_status = scanner.Scan();
   if (!scan_status.ok()) {
      std::cerr << "Error scanning CSV: " << scan_status.message()
                << std::endl;
      return 1;
   }

   return 0;
}
