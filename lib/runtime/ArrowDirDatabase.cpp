#include "runtime/ArrowDirDatabase.h"

#include <filesystem>
#include <fstream>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/status.h>
#include <arrow/table.h>

#include <random>
#include <ranges>

using namespace runtime;
std::shared_ptr<runtime::TableMetaData> runtime::ArrowDirDatabase::getTableMetaData(const std::string& name) {
   if (!metaData.contains(name)) {
      return runtime::TableMetaData::create("", name, {});
   }
   return metaData[name];
}

std::shared_ptr<arrow::Table> ArrowDirDatabase::loadTable(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   for (int i = 0; i < batchReader->num_record_batches(); i++) {
      batches.push_back(batchReader->ReadRecordBatch(i).ValueOrDie());
   }
   return arrow::Table::FromRecordBatches(batchReader->schema(), batches).ValueOrDie();
}
void storeTable(std::string file, std::shared_ptr<arrow::Table> table) {
   auto inputFile = arrow::io::FileOutputStream::Open(file).ValueOrDie();
   auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, table->schema()).ValueOrDie();
   assert(batchWriter->WriteTable(*table).ok());
   assert(batchWriter->Close().ok());
   assert(inputFile->Close().ok());
}
void storeSample(std::string file, std::shared_ptr<arrow::RecordBatch> table) {
   auto inputFile = arrow::io::FileOutputStream::Open(file).ValueOrDie();
   auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, table->schema()).ValueOrDie();
   assert(batchWriter->WriteRecordBatch(*table).ok());
   assert(batchWriter->Close().ok());
   assert(inputFile->Close().ok());
}
std::shared_ptr<arrow::RecordBatch> ArrowDirDatabase::loadSample(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   assert(batchReader->num_record_batches() == 1);
   auto batch = batchReader->ReadRecordBatch(0).ValueOrDie();
   return batch;
}

std::unique_ptr<Database> ArrowDirDatabase::load(std::string directory) {
   std::string json;
   auto database = std::make_unique<ArrowDirDatabase>();
   database->directory = directory;
   database->writeback = false;
   for (const auto& p : std::filesystem::directory_iterator(directory)) {
      auto path = p.path();
      if (path.extension().string() == ".arrow") {
         auto tablename = path.stem().string();
         database->tables[tablename] = loadTable(path.string());
      }
      if (path.extension().string() == ".sample" && path.stem().string().ends_with(".arrow")) {
         auto stem = path.stem().string();
         auto tablename = stem.substr(0, stem.size() - std::string(".arrow").size());
         database->samples[tablename] = loadSample(path.string());
      }
      if (path.filename() == "metadata.json") {
         std::ifstream t(path);
         json = std::string((std::istreambuf_iterator<char>(t)),
                            std::istreambuf_iterator<char>());
      }
   }
   for (auto& table : database->tables) {
      database->metaData[table.first] = runtime::TableMetaData::create(json, table.first, database->getSample(table.first));
   }
   return database;
}
std::shared_ptr<arrow::Table> ArrowDirDatabase::getTable(const std::string& name) {
   return tables[name];
}
std::shared_ptr<arrow::RecordBatch> ArrowDirDatabase::getSample(const std::string& name) {
   if (samples.contains(name)) {
      return samples[name];
   }
   return std::shared_ptr<arrow::RecordBatch>();
}
bool ArrowDirDatabase::hasTable(const std::string& name) {
   return tables.contains(name);
}
size_t asInt(std::variant<size_t, std::string> intOrStr) {
   if (std::holds_alternative<size_t>(intOrStr)) {
      return std::get<size_t>(intOrStr);
   } else {
      return std::stoll(std::get<std::string>(intOrStr));
   }
}
std::shared_ptr<arrow::DataType> createDataType(const ColumnType& columnType) {
   if (columnType.base == "bool") return arrow::boolean();
   if (columnType.base == "int") {
      switch (asInt(columnType.modifiers.at(0))) {
         case 8: return arrow::int8();
         case 16: return arrow::int16();
         case 32: return arrow::int32();
         case 64: return arrow::int64();
      }
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
   if (columnType.base == "decimal") return arrow::decimal(asInt(columnType.modifiers.at(0)), asInt(columnType.modifiers.at(1)));
   throw std::runtime_error("unsupported type");
}
std::shared_ptr<arrow::Schema> createSchema(std::shared_ptr<TableMetaData> metaData) {
   arrow::FieldVector fields;
   for (auto c : metaData->getOrderedColumns()) {
      auto& columnMetaData = metaData->getColumnMetaData(c);
      fields.push_back(std::make_shared<arrow::Field>(c, createDataType(columnMetaData->getColumnType())));
   }
   return std::make_shared<arrow::Schema>(fields);
}
void ArrowDirDatabase::createTable(std::string tableName, std::shared_ptr<TableMetaData> mD) {
   std::shared_ptr<arrow::Schema> schema = createSchema(mD);
   std::vector<std::shared_ptr<arrow::ChunkedArray>> cols;
   for (size_t i = 0; i < mD->getOrderedColumns().size(); i++) {
      cols.push_back(arrow::ChunkedArray::MakeEmpty(schema->field(i)->type()).ValueOrDie());
   }
   tables[tableName] = arrow::Table::Make(schema, cols);
   metaData[tableName] = mD;
}
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
   std::vector<int> result;
   arrow::NumericBuilder<arrow::Int32Type> numericBuilder;

   auto rng = std::mt19937{std::random_device{}()};

   // sample five values without replacement from [1, 100]
   std::sample(
      BoxedIntegerIterator{0l}, BoxedIntegerIterator{table->num_rows() - 1},
      std::back_inserter(result), std::min(table->num_rows(), 1024l), rng);
   for (auto i : result) {
      assert(numericBuilder.Append(i).ok());
   }
   auto indices = numericBuilder.Finish().ValueOrDie();
   std::vector<arrow::Datum> args({table, indices});

   auto res = arrow::compute::CallFunction("take", args).ValueOrDie();
   return res.table()->CombineChunksToBatch().ValueOrDie();
}
void ArrowDirDatabase::appendTable(std::string tableName, std::shared_ptr<arrow::Table> newRows) {
   if (!hasTable(tableName)) {
      throw std::runtime_error("can not append to non-existing table " + tableName);
   }
   auto table = tables.at(tableName);
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   if (table->num_rows() != 0) {
      batches.push_back(table->CombineChunksToBatch().ValueOrDie());
   }
   batches.push_back(newRows->CombineChunksToBatch().ValueOrDie());
   tables[tableName] = arrow::Table::FromRecordBatches(batches).ValueOrDie();
   samples[tableName] = createSample(tables[tableName]);
   metaData[tableName]->setNumRows(tables[tableName]->num_rows());
}
ArrowDirDatabase::~ArrowDirDatabase() {
   if (writeback) {
      writeMetaData(directory + "/metadata.json");
      for (auto t : tables) {
         storeTable(directory + "/" + t.first + ".arrow", t.second);
      }
      for (auto s : samples) {
         storeSample(directory + "/" + s.first + ".arrow.sample", s.second);
      }
   }
}
void ArrowDirDatabase::writeMetaData(std::string filename) {
   std::ofstream ostream(filename);
   ostream << "{ \"tables\": {";
   bool first = true;
   for (auto t : metaData) {
      if (first) {
         first = false;
      } else {
         ostream << ",";
      }
      ostream << "\"" << t.first << "\":" << t.second->serialize(false);
   }
   ostream << "} }";
}
void ArrowDirDatabase::setWriteback(bool writeback) {
   ArrowDirDatabase::writeback = writeback;
}
