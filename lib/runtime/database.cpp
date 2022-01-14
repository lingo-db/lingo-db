#include <filesystem>
#include <fstream>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <iostream>

#include "runtime/database.h"

namespace runtime {
void Database::addTable(std::string name, std::shared_ptr<arrow::Table> table) {
   tables[name] = table;
}

std::shared_ptr<arrow::Table> Database::loadTable(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   for (int i = 0; i < batchReader->num_record_batches(); i++) {
      batches.push_back(batchReader->ReadRecordBatch(i).ValueOrDie());
   }
   return arrow::Table::FromRecordBatches(batchReader->schema(), batches).ValueOrDie();
}
std::shared_ptr<arrow::RecordBatch> Database::loadSample(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   assert(batchReader->num_record_batches() == 1);
   auto batch = batchReader->ReadRecordBatch(0).ValueOrDie();
   return batch;
}

std::unique_ptr<Database> Database::load(std::string directory) {
   std::string json;
   auto database = std::make_unique<Database>();
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
std::shared_ptr<arrow::Table> Database::getTable(const std::string& name) {
   return tables[name];
}
std::shared_ptr<arrow::RecordBatch> Database::getSample(const std::string& name) {
   if (samples.contains(name)) {
      return samples[name];
   }
   return std::shared_ptr<arrow::RecordBatch>();
}
std::string Database::serializeRecordBatch(std::shared_ptr<arrow::RecordBatch> batch) {
   std::shared_ptr<arrow::ResizableBuffer> buffer = arrow::AllocateResizableBuffer(0).ValueOrDie();
   std::unique_ptr<arrow::io::BufferOutputStream> bufferOutputStream = std::make_unique<arrow::io::BufferOutputStream>(buffer);
   std::shared_ptr<arrow::ipc::RecordBatchWriter> recordBatchWriter = arrow::ipc::MakeStreamWriter(bufferOutputStream.get(), batch->schema()).ValueOrDie();

   assert(recordBatchWriter->WriteRecordBatch(*batch) == arrow::Status::OK());
   assert(recordBatchWriter->Close() == arrow::Status::OK());
   auto resBuffer = bufferOutputStream->Finish();
   return resBuffer.ValueOrDie()->ToHexString();
}
//adapted from stack overflow: https://stackoverflow.com/a/30606613
std::shared_ptr<arrow::ResizableBuffer> hexToBytes(const std::string& hex) {
   auto bytes = arrow::AllocateResizableBuffer(hex.size() / 2).ValueOrDie();
   for (unsigned int i = 0; i < hex.length(); i += 2) {
      std::string byteString = hex.substr(i, 2);
      char byte = (char) strtol(byteString.c_str(), NULL, 16);
      bytes->mutable_data()[i / 2] = byte;
   }

   return bytes;
}
std::shared_ptr<arrow::RecordBatch> Database::deserializeRecordBatch(std::string str) {
   auto resizableBuffer = hexToBytes(str);
   auto reader = arrow::ipc::RecordBatchStreamReader::Open(std::make_shared<arrow::io::BufferReader>(resizableBuffer)).ValueOrDie();
   std::shared_ptr<arrow::RecordBatch> batch;
   assert(reader->ReadNext(&batch) == arrow::Status::OK());
   return batch;
}
std::shared_ptr<TableMetaData> Database::getTableMetaData(const std::string& name) {
   if(!metaData.contains(name)) {
      return runtime::TableMetaData::create("", name, {});
   }
   return metaData[name];
}
} //end namespace runtime