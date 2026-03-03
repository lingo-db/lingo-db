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

class ArrowScanTask : public lingodb::scheduler::Task {
   public:
   ArrowScanTask() : taskQueue_() {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         currentTasks_.push_back(std::nullopt);
      }
   }
   struct ArrowTaskInfo {
      arrow::internal::FnOnce<void()> task;
      arrow::StopToken stop_token;
      arrow::internal::Executor::StopCallback stop_callback;
      arrow::internal::TaskHints hints;
      std::shared_ptr<ArrowTaskInfo> next;
   };

   struct ArrowTaskCompare {
      bool operator()(const std::shared_ptr<ArrowTaskInfo>& a,
                      const std::shared_ptr<ArrowTaskInfo>& b) const {
         const auto& ha = a->hints;
         const auto& hb = b->hints;

         // Arrow: lower priority value = more urgent
         if (ha.priority != hb.priority) return ha.priority > hb.priority;

         // Optional deterministic tie-breakers:
         if (ha.io_size != hb.io_size) return ha.io_size < hb.io_size; // larger IO first
         if (ha.cpu_cost != hb.cpu_cost) return ha.cpu_cost < hb.cpu_cost; // heavier CPU first
         return ha.external_id > hb.external_id; // stable-ish order
      }
   };

   public:
   void add_arrow_task(std::shared_ptr<ArrowTaskInfo> task_info) {
      std::unique_lock<std::mutex> lock(mutex_);
      taskQueue_.emplace(task_info);
   }
   bool allocateWork() override {
      std::unique_lock<std::mutex> lock(mutex_);
      if (taskQueue_.empty()) {
         return false;
      }
      auto top = taskQueue_.top();
      taskQueue_.pop();
      if (!top) {
         return false;
      }
      currentTasks_[lingodb::scheduler::currentWorkerId()] = std::move(top);
      return true;
   }
   void performWork() override {
      auto currentTask = currentTasks_[lingodb::scheduler::currentWorkerId()];
      if (!currentTask.has_value() || !currentTask.value()->task) return;
      if (!currentTask.value()->stop_token.IsStopRequested()) {
         std::move(currentTask.value()->task)();
      } else if (currentTask.value()->stop_callback) {
         std::move(currentTask.value()->stop_callback)(currentTask.value()->stop_token.Poll());
      }

      currentTasks_[lingodb::scheduler::currentWorkerId()] = std::nullopt;
   }
   void setup() override {
   }
   void teardown() override {
   }
   void finish() {
      workExhausted.store(true);
   }
   ~ArrowScanTask() override {
   }

   private:
   std::priority_queue<
      std::shared_ptr<ArrowTaskInfo>,
      std::vector<std::shared_ptr<ArrowTaskInfo>>,
      ArrowTaskCompare>
      taskQueue_;
   std::vector<std::optional<std::shared_ptr<ArrowTaskInfo>>> currentTasks_;
   std::mutex mutex_;
   std::condition_variable cv_;
};
class CustomExecutor : public arrow::internal::Executor {
   public:
   CustomExecutor(std::unique_ptr<lingodb::scheduler::SchedulerHandle> scheduler) : scheduler_(std::move(scheduler)) {
      auto masterTask = std::make_unique<ArrowScanTask>();
      master_task_ = masterTask.get();
      lingodb::scheduler::enqueueTask(std::move(masterTask));
   }
   arrow::Status SpawnReal(arrow::internal::TaskHints hints, arrow::internal::FnOnce<void()> task, arrow::StopToken stopToken,
                           StopCallback&& stop_callback) override {
      assert(master_task_);
      master_task_->add_arrow_task(std::make_shared<ArrowScanTask::ArrowTaskInfo>(std::move(task), stopToken, std::move(stop_callback), hints));
      return arrow::Status::OK();
   }

   int GetCapacity() override {
      return lingodb::scheduler::getNumWorkers();
   }

   private:
   std::unique_ptr<lingodb::scheduler::SchedulerHandle> scheduler_;

   public:
   ArrowScanTask* master_task_;
};

class CSVScanner {
   public:
   CSVScanner(const std::string& file_path) : file_path_(file_path) {}

   arrow::Result<size_t> Init() {
      // Open the CSV file
      ARROW_ASSIGN_OR_RAISE(auto input_file,
                            arrow::io::ReadableFile::Open(file_path_));
#if MULTITHREADED
      //Reason for slowdown
      auto scheduler = lingodb::scheduler::startScheduler();
      auto custom_executor = std::make_shared<CustomExecutor>(std::move(scheduler));
      custom_executor_ = custom_executor;
#endif

      auto csv_read_opts = arrow::csv::ReadOptions::Defaults();
      csv_read_opts.use_threads = MULTITHREADED;
      csv_read_opts.block_size = 1 << 20; // 1 MB blocks

      auto csv_parse_opts = arrow::csv::ParseOptions::Defaults();
      auto csv_convert_opts = arrow::csv::ConvertOptions::Defaults();
      auto init_start = std::chrono::high_resolution_clock::now();
#if MULTITHREADED
      auto io_context = arrow::io::IOContext(arrow::default_memory_pool());
      auto reader_future = arrow::csv::StreamingReader::MakeAsync(io_context,
                                                                  input_file,
                                                                  custom_executor.get(),
                                                                  csv_read_opts,
                                                                  csv_parse_opts,
                                                                  csv_convert_opts);

      ARROW_ASSIGN_OR_RAISE(auto reader, reader_future.result());
      reader_ = reader;
#if MULTITHREADED
      custom_executor_->master_task_->finish();
#endif

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
   std::shared_ptr<CustomExecutor> custom_executor_;
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
