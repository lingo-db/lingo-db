#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/runtime/RecordBatchInfo.h"
#include "lingodb/scheduler/Tasks.h"
#include "lingodb/utility/Serialization.h"
#include "lingodb/utility/Tracer.h"

#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>

#include <filesystem>
#include <iostream>
#include <random>
#include <ranges>

namespace {
namespace utility = lingodb::utility;
static utility::Tracer::Event processMorsel("DataSourceIteration", "processMorsel");

static utility::Tracer::Event processMorselSingle("DataSourceIteration", "processMorselSingle");

static utility::Tracer::Event cleanupTLS("DataSourceIteration", "cleanup");
std::vector<std::shared_ptr<arrow::RecordBatch>> loadTable(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   for (int i = 0; i < batchReader->num_record_batches(); i++) {
      batches.push_back(batchReader->ReadRecordBatch(i).ValueOrDie());
   }
   return batches;
}
void storeTable(std::string file, std::shared_ptr<arrow::Schema> schema, const std::vector<std::shared_ptr<arrow::RecordBatch>>& data) {
   auto inputFile = arrow::io::FileOutputStream::Open(file).ValueOrDie();
   auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, schema).ValueOrDie();
   for (auto& batch : data) {
      if (!batchWriter->WriteRecordBatch(*batch).ok()) {
         throw std::runtime_error("could not store table");
      }
   }
   auto close1 = batchWriter->Close();
   if (!close1.ok()) {
      throw std::runtime_error("could not store table:" + close1.ToString());
   }
   if (!inputFile->Close().ok()) {
      throw std::runtime_error("could not store table");
   }
}
/*
 * Create sample from arrow table
 */

template <typename I>
class BoxedIntegerIterator {
   I i;

   public:
   typedef I difference_type;
   typedef I value_type;
   typedef I pointer;
   typedef I reference;
   typedef std::random_access_iterator_tag iterator_category;

   BoxedIntegerIterator(I i) : i{i} {}

   bool operator==(BoxedIntegerIterator<I>& other) { return i == other.i; }
   I operator-(BoxedIntegerIterator<I>& other) { return i - other.i; }
   I operator++() { return i++; }
   I operator*() { return i; }
};
std::shared_ptr<arrow::RecordBatch> createSample(const std::vector<std::shared_ptr<arrow::RecordBatch>>& data) {
   size_t numRows = 0;
   for (auto& batch : data) {
      numRows += batch->num_rows();
   }
   if (numRows == 0) {
      return std::shared_ptr<arrow::RecordBatch>();
   }
   std::vector<size_t> result;

   auto rng = std::mt19937{std::random_device{}()};

   // sample five values without replacement from [1, 100]
   std::sample(
      BoxedIntegerIterator<size_t>{0l}, BoxedIntegerIterator<size_t>{numRows},
      std::back_inserter(result), std::min<size_t>(numRows, 1024ull), rng);
   std::sort(result.begin(), result.end());
   size_t currPos = 0;
   size_t currBatch = 0;
   size_t batchStart = 0;
   std::vector<std::shared_ptr<arrow::RecordBatch>> sampleData;
   while (currPos < result.size()) {
      std::vector<size_t> fromCurrentBatch;
      while (currPos < result.size() && result[currPos] < batchStart + data[currBatch]->num_rows()) {
         fromCurrentBatch.push_back(result[currPos] - batchStart);
         currPos++;
      }
      if (fromCurrentBatch.size() > 0) {
         arrow::NumericBuilder<arrow::Int32Type> numericBuilder;
         for (auto i : fromCurrentBatch) {
            if (!numericBuilder.Append(i).ok()) {
               throw std::runtime_error("could not create sample");
            }
         }
         auto indices = numericBuilder.Finish().ValueOrDie();
         std::vector<arrow::Datum> args({data[currBatch], indices});
         auto res = arrow::compute::CallFunction("take", args).ValueOrDie();
         sampleData.push_back(res.record_batch());
      }
      batchStart += data[currBatch]->num_rows();
      currBatch++;
   }
   return arrow::Table::FromRecordBatches(sampleData).ValueOrDie()->CombineChunksToBatch().ValueOrDie();
}

std::shared_ptr<arrow::DataType> toPhysicalType(lingodb::catalog::Type t) {
   using TypeId = lingodb::catalog::LogicalTypeId;
   switch (t.getTypeId()) {
      case TypeId::BOOLEAN:
         return arrow::boolean();
      case TypeId::INT:
         switch (t.getInfo<lingodb::catalog::IntTypeInfo>()->getBitWidth()) {
            case 8:
               return arrow::int8();
            case 16:
               return arrow::int16();
            case 32:
               return arrow::int32();
            case 64:
               return arrow::int64();
            default:
               throw std::runtime_error("unsupported bit width");
         }
      case TypeId::FLOAT:
         return arrow::float32();
      case TypeId::DOUBLE:
         return arrow::float64();
      case TypeId::DECIMAL:
         return arrow::decimal128(t.getInfo<lingodb::catalog::DecimalTypeInfo>()->getPrecision(),
                                  t.getInfo<lingodb::catalog::DecimalTypeInfo>()->getScale());
      case TypeId::DATE: {
         auto dateUnit = t.getInfo<lingodb::catalog::DateTypeInfo>()->getUnit();
         switch (dateUnit) {
            case lingodb::catalog::DateTypeInfo::DateUnit::DAY:
               return arrow::date32();
            case lingodb::catalog::DateTypeInfo::DateUnit::MILLIS:
               return arrow::date64();
         }
      }
      case TypeId::TIMESTAMP: {
         arrow::TimeUnit::type timeUnit;
         auto logicalTimeUnit = t.getInfo<lingodb::catalog::TimestampTypeInfo>()->getUnit();
         switch (logicalTimeUnit) {
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::NANOS:
               timeUnit = arrow::TimeUnit::NANO;
               break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::MICROS:
               timeUnit = arrow::TimeUnit::MICRO;
               break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::MILLIS:
               timeUnit = arrow::TimeUnit::MILLI;
               break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::SECONDS:
               timeUnit = arrow::TimeUnit::SECOND;
               break;
         }
         return arrow::timestamp(timeUnit);
      }
      case TypeId::INTERVAL: {
         auto intervalUnit = t.getInfo<lingodb::catalog::IntervalTypeInfo>()->getUnit();
         switch (intervalUnit) {
            case lingodb::catalog::IntervalTypeInfo::IntervalUnit::DAYTIME:
               return arrow::day_time_interval();
            case lingodb::catalog::IntervalTypeInfo::IntervalUnit::MONTH:
               return arrow::month_interval();
         }
      }
      case TypeId::CHAR:
         return arrow::fixed_size_binary(t.getInfo<lingodb::catalog::CharTypeInfo>()->getLength());
      case TypeId::STRING:
         return arrow::utf8();
      default:
         throw std::runtime_error("unsupported type");
   }
}
std::optional<size_t> countDistinctValues(std::shared_ptr<arrow::ChunkedArray> column) {
   //todo: replace with approximate count in the future
   auto res = arrow::compute::CallFunction("count_distinct", {column});
   if (res.ok()) {
      return res.ValueOrDie().scalar_as<arrow::Int64Scalar>().value;
   }
   return {};
}



} // namespace

namespace lingodb::runtime {

std::unique_ptr<LingoDBTable> LingoDBTable::create(const catalog::CreateTableDef& def) {
   arrow::FieldVector fields;
   for (auto c : def.columns) {
      fields.push_back(std::make_shared<arrow::Field>(c.getColumnName(), toPhysicalType(c.getLogicalType())));
   }
   auto arrowSchema = std::make_shared<arrow::Schema>(fields);
   return std::make_unique<LingoDBTable>(def.name + ".arrow", arrowSchema);
}
LingoDBTable::LingoDBTable(std::string fileName, std::shared_ptr<arrow::Schema> arrowSchema) : persist(false), fileName(std::move(fileName)), sample(arrowSchema), schema(std::move(arrowSchema)), tableData(), numRows(0) {
   for (auto c : schema->fields()) {
      columnStatistics[c->name()] = catalog::ColumnStatistics(std::nullopt);
   }
}
void LingoDBTable::append(const std::shared_ptr<arrow::Table>& table) {
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   arrow::TableBatchReader reader(table);
   std::shared_ptr<arrow::RecordBatch> nextChunk;
   while (reader.ReadNext(&nextChunk) == arrow::Status::OK()) {
      if (nextChunk) {
         batches.push_back(nextChunk);
      } else {
         break;
      }
      nextChunk.reset();
   }
   append(batches);
}
void LingoDBTable::append(const std::vector<std::shared_ptr<arrow::RecordBatch>>& toAppend) {
   ensureLoaded();
   for (auto& batch : toAppend) {
      if (batch->schema()->Equals(*schema)) {
         tableData.push_back(batch);
         numRows += batch->num_rows();
      } else {
         std::cout << "schema to add: " << batch->schema()->ToString() << std::endl;
         std::cout << "schema of table: " << schema->ToString() << std::endl;
         throw std::runtime_error("schema mismatch");
      }
   }
   sample = createSample(tableData);
   auto tableView = arrow::Table::FromRecordBatches(toAppend).ValueOrDie();
   for (auto c : schema->fields()) {
      columnStatistics[c->name()] = catalog::ColumnStatistics(countDistinctValues(tableView->GetColumnByName(c->name())));
   }
   flush();
}
const catalog::ColumnStatistics& LingoDBTable::getColumnStatistics(std::string column) const {
   if (!columnStatistics.contains(column)) {
      throw std::runtime_error("MetaData: Column not found");
   }
   return columnStatistics.at(column);
}
void LingoDBTable::flush() {
   if (!persist) return;
   ensureLoaded();
   storeTable(dbDir + "/" + fileName, schema, tableData);
}

std::shared_ptr<arrow::DataType> LingoDBTable::getColumnStorageType(const std::string& columnName) const {
   auto field = schema->GetFieldByName(columnName);
   if (!field) {
      throw std::runtime_error("column not found");
   }
   return field->type();
}
void LingoDBTable::ensureLoaded() {
   if (!loaded) {
      loaded = true;
      if (fileName.empty() || dbDir.empty()) {
         return;
      }
      if (!std::filesystem::exists(dbDir + "/" + fileName)) {
         return;
      }
      tableData = loadTable(dbDir + "/" + fileName);
   }
}
void LingoDBTable::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, fileName);
   serializer.writeProperty(2, sample);
   auto res = arrow::ipc::SerializeSchema(*schema).ValueOrDie();
   serializer.writeProperty(3, std::string_view((const char*) res->data(), res->size()));
   serializer.writeProperty(4, columnStatistics);
   serializer.writeProperty(5, numRows);
}
std::unique_ptr<LingoDBTable> LingoDBTable::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto fileName = deserializer.readProperty<std::string>(1);
   auto sample = deserializer.readProperty<catalog::Sample>(2);
   auto schemaData = deserializer.readProperty<std::string>(3);
   arrow::ipc::DictionaryMemo dictMemo;
   auto bufferReader = arrow::io::BufferReader::FromString(schemaData);
   auto schema = arrow::ipc::ReadSchema(bufferReader.get(), &dictMemo).ValueOrDie();
   auto columnStatistics = deserializer.readProperty<std::unordered_map<std::string, catalog::ColumnStatistics>>(4);
   auto numRows = deserializer.readProperty<size_t>(5);
   return std::make_unique<LingoDBTable>(fileName, schema, numRows, sample, columnStatistics);
}

class ScanBatchesTask : public lingodb::scheduler::TaskWithImplicitContext {
   std::vector<std::shared_ptr<arrow::RecordBatch>>& batches;
   std::vector<size_t> colIds;
   std::function<void(lingodb::runtime::RecordBatchInfo*)> cb;
   std::atomic<size_t> startIndex{0};
   std::vector<lingodb::runtime::RecordBatchInfo*> batchInfos;
   std::vector<size_t> workerResvs;

   public:
   ScanBatchesTask(std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, std::vector<size_t> colIds, const std::function<void(lingodb::runtime::RecordBatchInfo*)>& cb) : batches(batches), colIds(colIds), cb(cb) {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         batchInfos.push_back(reinterpret_cast<lingodb::runtime::RecordBatchInfo*>(malloc(sizeof(lingodb::runtime::RecordBatchInfo) + sizeof(lingodb::runtime::ColumnInfo) * colIds.size())));
         workerResvs.push_back(0);
      }
   }

   bool allocateWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex >= batches.size()) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }
   void performWork() override {
      auto& batch = batches[workerResvs[lingodb::scheduler::currentWorkerId()]];
      auto* batchInfo = batchInfos[lingodb::scheduler::currentWorkerId()];
      utility::Tracer::Trace trace(processMorsel);
      RecordBatchInfo::access(colIds, batchInfo, batch);
      cb(batchInfo);
      trace.stop();
   }
   ~ScanBatchesTask() {
      utility::Tracer::Trace cleanUpTrace(cleanupTLS);
      for (auto* bI : batchInfos) {
         if (bI) {
            free(bI);
         }
      }
      cleanUpTrace.stop();
   }
};
class ScanBatchesSingleThreadedTask : public lingodb::scheduler::TaskWithImplicitContext {
   std::vector<std::shared_ptr<arrow::RecordBatch>>& batches;
   std::vector<size_t> colIds;
   std::function<void(lingodb::runtime::RecordBatchInfo*)> cb;

   public:
   ScanBatchesSingleThreadedTask(std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, std::vector<size_t> colIds, const std::function<void(lingodb::runtime::RecordBatchInfo*)>& cb) : batches(batches), colIds(colIds), cb(cb) {
   }

   bool allocateWork() override {
      if (!workExhausted.exchange(true)) {
         return true;
      }
      return false;
   }
   void performWork() override {
      auto* batchInfo = reinterpret_cast<lingodb::runtime::RecordBatchInfo*>(malloc(sizeof(lingodb::runtime::RecordBatchInfo) + sizeof(lingodb::runtime::ColumnInfo) * colIds.size()));

      for (const auto& batch : batches) {
         utility::Tracer::Trace trace(processMorselSingle);
         RecordBatchInfo::access(colIds, batchInfo, batch);
         cb(batchInfo);
         trace.stop();
      }
      free(batchInfo);
   }
   ~ScanBatchesSingleThreadedTask() {
   }
};

std::unique_ptr<scheduler::Task> LingoDBTable::createScanTask(const ScanConfig& scanConfig) {
   ensureLoaded();
   std::vector<size_t> colIds;
   for (const auto& c : scanConfig.columns) {
      auto colId = schema->GetFieldIndex(c);
      assert(colId >= 0);
      colIds.push_back(colId);
   }
   if (scanConfig.parallel) {
      return std::make_unique<ScanBatchesTask>(tableData, colIds, scanConfig.cb);
   } else {
      return std::make_unique<ScanBatchesSingleThreadedTask>(tableData, colIds, scanConfig.cb);
   }
}

std::pair<std::shared_ptr<arrow::RecordBatch>, size_t> LingoDBTable::getByRowId(size_t rowId) const {
   size_t currRow = 0;
   for (const auto& batch : tableData) {
      if (currRow + batch->num_rows() > rowId) {
         return {batch, rowId - currRow};
      }
      currRow += batch->num_rows();
   }
   throw std::runtime_error("row id out of bounds");
}

size_t LingoDBTable::getColIndex(std::string colName) {
   return schema->GetFieldIndex(colName);
}

} // namespace lingodb::runtime