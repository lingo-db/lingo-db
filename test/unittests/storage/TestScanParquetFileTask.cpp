#include "catch2/catch_all.hpp"
#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/storage/ExternalTableScan.h"
#include "lingodb/runtime/storage/Restrictions.h"
#include "lingodb/runtime/storage/TableStorage.h"

#include <arrow/builder.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <memory>
#include <vector>

using namespace lingodb::runtime;
namespace fs = std::filesystem;

namespace {

class TestParquetFileGenerator {
   public:
   static std::string createTestParquetFile(const std::string& fileName,
                                            const std::vector<int64_t>& intValues,
                                            const std::vector<std::string>& stringValues,
                                            int64_t rowGroupSize = 1) {
      auto schema = arrow::schema({arrow::field("int_col", arrow::int64()),
                                   arrow::field("string_col", arrow::utf8())});

      arrow::Int64Builder intBuilder;
      for (int64_t val : intValues) {
         REQUIRE(intBuilder.Append(val).ok());
      }
      auto intArray = intBuilder.Finish().ValueOrDie();

      arrow::StringBuilder stringBuilder;
      for (const auto& str : stringValues) {
         REQUIRE(stringBuilder.Append(str).ok());
      }
      auto stringArray = stringBuilder.Finish().ValueOrDie();

      auto batch = arrow::RecordBatch::Make(schema, intValues.size(), {intArray, stringArray});
      auto table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();

      std::string filePath = std::filesystem::temp_directory_path().string() + "/" + fileName;
      auto outfile = arrow::io::FileOutputStream::Open(filePath).ValueOrDie();

      REQUIRE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, rowGroupSize).ok());
      REQUIRE(outfile->Close().ok());

      return filePath;
   }

   static std::string createTestParquetFileWithNulls(const std::string& fileName,
                                                     const std::vector<int64_t>& intValues,
                                                     const std::vector<bool>& intNulls,
                                                     const std::vector<std::string>& stringValues,
                                                     const std::vector<bool>& stringNulls,
                                                     int64_t rowGroupSize = 1) {
      auto schema = arrow::schema({arrow::field("int_col", arrow::int64()),
                                   arrow::field("string_col", arrow::utf8())});

      arrow::Int64Builder intBuilder;
      for (size_t i = 0; i < intValues.size(); ++i) {
         if (intNulls[i]) {
            REQUIRE(intBuilder.AppendNull().ok());
         } else {
            REQUIRE(intBuilder.Append(intValues[i]).ok());
         }
      }
      auto intArray = intBuilder.Finish().ValueOrDie();

      arrow::StringBuilder stringBuilder;
      for (size_t i = 0; i < stringValues.size(); ++i) {
         if (stringNulls[i]) {
            REQUIRE(stringBuilder.AppendNull().ok());
         } else {
            REQUIRE(stringBuilder.Append(stringValues[i]).ok());
         }
      }
      auto stringArray = stringBuilder.Finish().ValueOrDie();

      auto batch = arrow::RecordBatch::Make(schema, intValues.size(), {intArray, stringArray});
      auto table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();

      std::string filePath = std::filesystem::temp_directory_path().string() + "/" + fileName;
      auto outfile = arrow::io::FileOutputStream::Open(filePath).ValueOrDie();

      REQUIRE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, rowGroupSize).ok());
      REQUIRE(outfile->Close().ok());

      return filePath;
   }
};

class MockTaskWithContext : public lingodb::scheduler::TaskWithContext {
   std::function<void()> job;

   public:
   MockTaskWithContext(lingodb::runtime::ExecutionContext* context, std::function<void()> job) : TaskWithContext(context), job(std::move(job)) {}
   bool allocateWork() override {
      if (workExhausted.exchange(true)) {
         return false;
      }
      return true;
   }
   void performWork() override {
      job();
   }
};

template <class F>
void withExecutionContext(const F& f) {
   auto scheduler = lingodb::scheduler::startScheduler();
   auto session = lingodb::runtime::Session::createSession();
   auto context = session->createExecutionContext();
   lingodb::scheduler::awaitEntryTask(std::make_unique<MockTaskWithContext>(context.get(), [&]() {
      f();
   }));
}

std::vector<int64_t> runScanAndCollectValues(const std::string& filePath, std::vector<FilterDescription> filterDescs, int64_t rowGroupSize = 20) {
   std::vector<int64_t> scannedValues;
   withExecutionContext([&]() {
      auto schema = arrow::schema({arrow::field("int_col", arrow::int64()),
                                   arrow::field("string_col", arrow::utf8())});

      auto restrictions = Restrictions::create(filterDescs, *schema);
      REQUIRE(restrictions != nullptr);

      std::vector<int> colIds = {0, 1};
      auto callback = [&](BatchView* batch) {
         REQUIRE(batch != nullptr);
         REQUIRE(batch->arrays != nullptr);
         REQUIRE(batch->selectionVector != nullptr);
         auto* intCol = batch->arrays[0];
         REQUIRE(intCol != nullptr);
         REQUIRE(intCol->buffers[1] != nullptr);

         const auto* data = reinterpret_cast<const int64_t*>(intCol->buffers[1]);
         for (size_t i = 0; i < static_cast<size_t>(batch->length); ++i) {
            scannedValues.push_back(data[batch->selectionVector[i]]);
         }
      };

      ScanParquetFileTask task(filePath, colIds, callback, std::move(restrictions), filterDescs);

      while (task.allocateWork()) {
         task.performWork();
      }
   });
   return scannedValues;
}

} // namespace

TEST_CASE("ScanParquetFileTask filters - MinMaxMetadataFilter", "[storage][parquet][filters]") {
   std::vector<int64_t> intValues;
   for (int64_t i = 1; i <= 100; ++i) {
      intValues.push_back(i);
   }
   std::vector<std::string> stringValues(100, "test");

   std::string filePath = TestParquetFileGenerator::createTestParquetFile(
      "test_minmax_filter.parquet", intValues, stringValues, 100);

   auto filter1 = std::make_shared<MinMaxMetadataFilter<int64_t>>(10, 50, true, true);
   auto filter2 = std::make_shared<MinMaxMetadataFilter<int64_t>>(100, 200, false, true);
   auto filter3 = std::make_shared<MinMaxMetadataFilter<int64_t>>(80, 120, false, true);

   auto input = arrow::io::ReadableFile::Open(filePath).ValueOrDie();
   auto reader = parquet::ParquetFileReader::Open(input);

   auto metadata = reader->metadata();
   REQUIRE(metadata->num_row_groups() > 0);

   auto rgMeta = metadata->RowGroup(0);
   auto colMeta = rgMeta->ColumnChunk(0);
   auto stats = colMeta->statistics();

   REQUIRE(filter1->apply(stats) == true);
   REQUIRE(filter2->apply(stats) == false);
   REQUIRE(filter3->apply(stats) == true);

   fs::remove(filePath);
}

TEST_CASE("ScanParquetFileTask filters - NotNullFilter", "[storage][parquet][filters]") {
   std::vector<int64_t> intValues = {1, 2, 3, 4, 5};
   std::vector<bool> intNulls = {false, false, true, false, false};
   std::vector<std::string> stringValues = {"a", "b", "c", "d", "e"};
   std::vector<bool> stringNulls = {false, false, false, false, false};

   std::string filePath = TestParquetFileGenerator::createTestParquetFileWithNulls(
      "test_notnull_filter.parquet", intValues, intNulls, stringValues, stringNulls);

   auto filter = std::make_shared<NotNullFilter>();

   auto input = arrow::io::ReadableFile::Open(filePath).ValueOrDie();
   auto reader = parquet::ParquetFileReader::Open(input);

   auto metadata = reader->metadata();
   auto rgMeta = metadata->RowGroup(0);
   auto colMeta = rgMeta->ColumnChunk(0);
   auto stats = colMeta->statistics();

   REQUIRE(filter->apply(stats) == true);

   fs::remove(filePath);
}

TEST_CASE("ScanParquetFileTask applies multiple metadata filters", "[storage][parquet][filters]") {
   std::vector<int64_t> intValues;
   for (int64_t i = 1; i <= 100; ++i) {
      intValues.push_back(i);
   }
   std::vector<std::string> stringValues(100, "test");

   std::string filePath = TestParquetFileGenerator::createTestParquetFile(
      "test_range_filter_scan.parquet", intValues, stringValues, 20);

   std::vector<FilterDescription> filterDescs;
   FilterDescription lower;
   lower.columnName = "int_col";
   lower.columnId = 0;
   lower.op = FilterOp::GTE;
   lower.value = int64_t(15);
   filterDescs.push_back(lower);

   FilterDescription upper;
   lower.columnName = "int_col";
   lower.columnId = 0;
   lower.op = FilterOp::LT;
   lower.value = int64_t(25);
   filterDescs.push_back(lower);

   auto scannedValues = runScanAndCollectValues(filePath, filterDescs, 20);
   REQUIRE(scannedValues.size() == 10);
   REQUIRE(scannedValues.front() == 15);
   REQUIRE(scannedValues.back() == 24);

   fs::remove(filePath);
}

TEST_CASE("ScanParquetFileTask returns no rows", "[storage][parquet][filters]") {
   std::vector<int64_t> intValues;
   for (int64_t i = 1; i <= 100; ++i) {
      intValues.push_back(i);
   }
   std::vector<std::string> stringValues(100, "test");

   std::string filePath = TestParquetFileGenerator::createTestParquetFile(
      "test_no_match_filter_scan.parquet", intValues, stringValues, 20);

   std::vector<FilterDescription> filterDescs;
   FilterDescription filter;
   filter.columnName = "int_col";
   filter.columnId = 0;
   filter.op = FilterOp::GTE;
   filter.value = int64_t(200);
   filterDescs.push_back(filter);

   auto scannedValues = runScanAndCollectValues(filePath, filterDescs);
   REQUIRE(scannedValues.empty());

   fs::remove(filePath);
}
