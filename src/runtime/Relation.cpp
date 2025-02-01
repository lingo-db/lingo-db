#include "lingodb/runtime/Relation.h"
#include "lingodb/runtime/HashIndex.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/status.h>
#include <arrow/table.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <ranges>
namespace {
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
std::shared_ptr<arrow::RecordBatch> createSample(std::shared_ptr<arrow::Table> table) {
   if (table->num_rows() == 0) {
      return std::shared_ptr<arrow::RecordBatch>();
   }
   std::vector<int> result;
   arrow::NumericBuilder<arrow::Int32Type> numericBuilder;

   auto rng = std::mt19937{std::random_device{}()};

   // sample five values without replacement from [1, 100]
   std::sample(
      BoxedIntegerIterator<long>{0l}, BoxedIntegerIterator<long>{table->num_rows() - 1},
      std::back_inserter(result), std::min(table->num_rows(), 1024l), rng);
   for (auto i : result) {
      if(!numericBuilder.Append(i).ok()){
         throw std::runtime_error("could not create sample");
      }
   }
   auto indices = numericBuilder.Finish().ValueOrDie();
   std::vector<arrow::Datum> args({table, indices});

   auto res = arrow::compute::CallFunction("take", args).ValueOrDie();
   return res.table()->CombineChunksToBatch().ValueOrDie();
}

/*
 * Count distinct number of values in column
 */
std::optional<size_t> countDistinctValues(std::shared_ptr<arrow::ChunkedArray> column) {
   //todo: replace with approximate count in the future
   auto res = arrow::compute::CallFunction("count_distinct", {column});
   if (res.ok()) {
      return res.ValueOrDie().scalar_as<arrow::Int64Scalar>().value;
   }
   return {};
}
//loading table
std::shared_ptr<arrow::Table> loadTable(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   for (int i = 0; i < batchReader->num_record_batches(); i++) {
      batches.push_back(batchReader->ReadRecordBatch(i).ValueOrDie());
   }
   return arrow::Table::FromRecordBatches(batchReader->schema(), batches).ValueOrDie();
}
//splitting table into "good-sized chunks"
std::vector<std::shared_ptr<arrow::RecordBatch>> toRecordBatches(std::shared_ptr<arrow::Table> table) {
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   arrow::TableBatchReader reader(table);
   reader.set_chunksize(20000);
   std::shared_ptr<arrow::RecordBatch> nextChunk;
   while (reader.ReadNext(&nextChunk) == arrow::Status::OK()) {
      if (nextChunk) {
         batches.push_back(nextChunk);
      } else {
         break;
      }
      nextChunk.reset();
   }
   return batches;
}
//load sample:
std::shared_ptr<arrow::RecordBatch> loadSample(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   assert(batchReader->num_record_batches() == 1);
   auto batch = batchReader->ReadRecordBatch(0).ValueOrDie();
   return batch;
}
//create arrow schema
size_t asInt(std::variant<size_t, std::string> intOrStr) {
   if (std::holds_alternative<size_t>(intOrStr)) {
      return std::get<size_t>(intOrStr);
   } else {
      return std::stoll(std::get<std::string>(intOrStr));
   }
}

std::shared_ptr<arrow::DataType> createDataType(const lingodb::runtime::ColumnType& columnType) {
   if (columnType.base == "bool") return arrow::boolean();
   if (columnType.base == "int") {
      switch (asInt(columnType.modifiers.at(0))) {
         case 8: return arrow::int8();
         case 16: return arrow::int16();
         case 32: return arrow::int32();
         case 64: return arrow::int64();
      }
   }
   if (columnType.base == "index") {
      return arrow::int64();
   }
   if (columnType.base == "float") {
      switch (asInt(columnType.modifiers.at(0))) {
         case 16: return arrow::float16();
         case 32: return arrow::float32();
         case 64: return arrow::float64();
      }
   }
   if (columnType.base == "date") {
      return std::get<std::string>(columnType.modifiers.at(0)) == "day" ?
         arrow::date32() :
         arrow::date64();
   }
   if (columnType.base == "string") return arrow::utf8();
   if (columnType.base == "char") return arrow::fixed_size_binary(asInt(columnType.modifiers.at(0)));
   if (columnType.base == "decimal") return arrow::decimal128(asInt(columnType.modifiers.at(0)), asInt(columnType.modifiers.at(1)));
   if (columnType.base == "timestamp") return arrow::timestamp(arrow::TimeUnit::SECOND);
   throw std::runtime_error("unsupported type");
}
std::shared_ptr<arrow::Schema> createSchema(std::shared_ptr<lingodb::runtime::TableMetaData> metaData) {
   arrow::FieldVector fields;
   for (auto c : metaData->getOrderedColumns()) {
      auto& columnMetaData = metaData->getColumnMetaData(c);
      fields.push_back(std::make_shared<arrow::Field>(c, createDataType(columnMetaData->getColumnType())));
   }
   return std::make_shared<arrow::Schema>(fields);
}

//storing tables
void storeTable(std::string file, std::shared_ptr<arrow::Table> table) {
   auto inputFile = arrow::io::FileOutputStream::Open(file).ValueOrDie();
   auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, table->schema()).ValueOrDie();
   if(!batchWriter->WriteTable(*table).ok()||!batchWriter->Close().ok()||!inputFile->Close().ok()){
      throw std::runtime_error("could not store table");
   }
}
void storeSample(std::string file, std::shared_ptr<arrow::RecordBatch> batch) {
   auto inputFile = arrow::io::FileOutputStream::Open(file).ValueOrDie();
   auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, batch->schema()).ValueOrDie();
   if(!batchWriter->WriteRecordBatch(*batch).ok()||!batchWriter->Close().ok()||!inputFile->Close().ok()){
      throw std::runtime_error("could not store table");
   }
}
} // end namespace

namespace lingodb::runtime {
class DBRelation : public Relation {
   std::shared_ptr<arrow::Table> table;
   std::shared_ptr<TableMetaData> metaData;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches;
   std::shared_ptr<arrow::Schema> schema;
   std::shared_ptr<arrow::RecordBatch> sample;
   std::unordered_map<std::string, std::shared_ptr<Index>> indices;
   std::string dbDir;
   bool eagerLoading;
   void flush() {
      if (!persist) return;
      auto dataFile = dbDir + "/" + name + ".arrow";
      auto sampleFile = dbDir + "/" + name + ".arrow.sample";
      auto metaFile = dbDir + "/" + name + ".metadata.json";

      std::ofstream ostream(metaFile);
      ostream << metaData->serialize(false);
      ostream.flush();

      if (table) {
         storeTable(dataFile, table);
      }
      if (sample) {
         storeSample(sampleFile, sample);
      }
   }

   public:
   DBRelation(const std::string dbDir, const std::string name, const std::shared_ptr<arrow::Table>& table, const std::shared_ptr<TableMetaData>& metaData, const std::vector<std::shared_ptr<arrow::RecordBatch>>& recordBatches, const std::shared_ptr<arrow::Schema>& schema, const std::shared_ptr<arrow::RecordBatch>& sample, bool eagerLoading) : table(table), metaData(metaData), recordBatches(recordBatches), schema(schema), sample(sample), dbDir(dbDir), eagerLoading(eagerLoading) {
      Relation::name = name;
      for (auto index : metaData->getIndices()) {
         indices.insert({index->name, Index::createHashIndex(*index, *this, dbDir)});
      }
      for (auto idx : indices) {
         if (eagerLoading) {
            idx.second->ensureLoaded();
         }
         idx.second->setPersist(persist);
      }
   }

   std::shared_ptr<TableMetaData> getMetaData() override {
      return metaData;
   }
   std::shared_ptr<arrow::RecordBatch> getSample() override {
      return sample;
   }
   std::shared_ptr<arrow::Schema> getArrowSchema() override {
      return schema;
   }
   const std::vector<std::shared_ptr<arrow::RecordBatch>>& getRecordBatches() override {
      return recordBatches;
   }
   std::shared_ptr<Index> getIndex(const std::string name) override {
      if (indices.contains(name)) {
         return indices.at(name);
      }
      throw std::runtime_error("index not found");
   }
   void loadData() override {
      if (!eagerLoading && table->num_rows() == 0) {
         auto dataFile = dbDir + "/" + name + ".arrow";
         if (std::filesystem::exists(dataFile)) {
            table = loadTable(dataFile);
            recordBatches = toRecordBatches(table);
         }
      }
   }
   void append(std::shared_ptr<arrow::Table> toAppend) override {
      std::vector<std::shared_ptr<arrow::RecordBatch>> newTableBatches;

      if (table->num_rows() != 0) {
         newTableBatches.push_back(table->CombineChunksToBatch().ValueOrDie());
      }
      newTableBatches.push_back(toAppend->CombineChunksToBatch().ValueOrDie());
      table = arrow::Table::FromRecordBatches(newTableBatches).ValueOrDie();
      recordBatches = toRecordBatches(table);
      sample = createSample(table);
      metaData->setNumRows(table->num_rows());
      for (auto c : metaData->getOrderedColumns()) {
         metaData->getColumnMetaData(c)->setDistinctValues(countDistinctValues(table->GetColumnByName(c)));
      }
      flush();
      for (auto idx : indices) {
         idx.second->appendRows(toAppend);
      }
   }
   std::shared_ptr<arrow::Table> getTable() override {
      return table;
   }
   void setPersist(bool persist) override {
      Relation::setPersist(persist);
      flush();
      for (auto idx : indices) {
         idx.second->setPersist(persist);
      }
   }
};
std::shared_ptr<Relation> Relation::loadRelation(std::string dbDir, std::string name, std::string json, bool eagerLoading) {
   std::shared_ptr<TableMetaData> metaData;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches;
   std::shared_ptr<arrow::Schema> schema;
   std::shared_ptr<arrow::RecordBatch> sample;
   auto dataFile = dbDir + "/" + name + ".arrow";
   auto sampleFile = dbDir + "/" + name + ".arrow.sample";
   std::shared_ptr<arrow::Table> table;
   if (std::filesystem::exists(dataFile) && eagerLoading) {
      table = loadTable(dataFile);
      recordBatches = toRecordBatches(table);
      schema = table->schema();
   }
   if (std::filesystem::exists(sampleFile)) {
      sample = loadSample(sampleFile);
   }
   metaData = lingodb::runtime::TableMetaData::create(json, name, sample);
   if (!table) {
      schema = createSchema(metaData);
      table = arrow::Table::MakeEmpty(schema).ValueOrDie();
   }
   return std::make_shared<DBRelation>(dbDir, name, table, metaData, recordBatches, schema, sample, eagerLoading);
}
std::shared_ptr<Relation> Relation::createDBRelation(std::string dbDir, std::string name, std::shared_ptr<TableMetaData> metaData) {
   auto schema = createSchema(metaData);
   std::shared_ptr<arrow::RecordBatch> sample;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches;
   if (!metaData->getPrimaryKey().empty()) {
      auto pkIndex = std::make_shared<IndexMetaData>();
      pkIndex->type = Index::Type::HASH;
      pkIndex->name = "pk_hash";
      pkIndex->columns = metaData->getPrimaryKey();
      metaData->getIndices().push_back(pkIndex);
   }
   return std::make_shared<DBRelation>(dbDir, name, arrow::Table::MakeEmpty(schema).ValueOrDie(), metaData, recordBatches, schema, sample, true);
}

class LocalRelation : public Relation {
   std::shared_ptr<arrow::Table> table;
   std::shared_ptr<TableMetaData> metaData;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches;
   std::shared_ptr<arrow::Schema> schema;
   std::shared_ptr<arrow::RecordBatch> sample;

   public:
   LocalRelation(std::shared_ptr<arrow::Table> table, std::shared_ptr<TableMetaData> metaData) : table(table), metaData(metaData) {
      sample = createSample(table);
      schema = createSchema(metaData);
      recordBatches = toRecordBatches(table);
   }
   LocalRelation(std::shared_ptr<TableMetaData> metaData) : metaData(metaData) {
      schema = createSchema(metaData);
      table = arrow::Table::MakeEmpty(schema).ValueOrDie();
   }
   std::shared_ptr<TableMetaData> getMetaData() override {
      return metaData;
   }
   std::shared_ptr<arrow::RecordBatch> getSample() override {
      return sample;
   }
   std::shared_ptr<arrow::Schema> getArrowSchema() override {
      return schema;
   }
   const std::vector<std::shared_ptr<arrow::RecordBatch>>& getRecordBatches() override {
      return recordBatches;
   }
   std::shared_ptr<Index> getIndex(const std::string name) override {
      throw std::runtime_error("indexes are not supported");
   }
   std::shared_ptr<arrow::Table> getTable() override {
      return table;
   }
   void loadData() override {
      //no effect
   }

   void append(std::shared_ptr<arrow::Table> toAppend) override {
      std::vector<std::shared_ptr<arrow::RecordBatch>> newTableBatches;

      if (table->num_rows() != 0) {
         newTableBatches.push_back(table->CombineChunksToBatch().ValueOrDie());
      }
      newTableBatches.push_back(toAppend->CombineChunksToBatch().ValueOrDie());
      table = arrow::Table::FromRecordBatches(newTableBatches).ValueOrDie();
      recordBatches = toRecordBatches(table);
      sample = createSample(table);
      metaData->setNumRows(table->num_rows());
   }
};
std::shared_ptr<Relation> Relation::createLocalRelation(std::string name, std::shared_ptr<TableMetaData> metaData) {
   return std::make_shared<LocalRelation>(metaData);
}

} // end namespace lingodb::runtime
