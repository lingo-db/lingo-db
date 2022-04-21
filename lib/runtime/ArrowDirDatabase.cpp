#include "runtime/ArrowDirDatabase.h"

#include <filesystem>
#include <fstream>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/status.h>
#include <arrow/table.h>

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
