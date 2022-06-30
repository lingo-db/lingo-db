#include <filesystem>
#include <fstream>
#include <iostream>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/status.h>

#include "runtime/ArrowDirDatabase.h"
#include "runtime/Database.h"

namespace runtime {
void Database::createTable(std::string tableName, std::shared_ptr<TableMetaData>) {
   throw std::runtime_error("database does not support table creation");
}
void Database::appendTable(std::string tableName, std::shared_ptr<arrow::Table> newRows) {
   throw std::runtime_error("database does not support table append");
}
std::unique_ptr<Database> Database::loadFromDir(std::string directory) {
   return ArrowDirDatabase::load(directory);
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

void Database::createTable(runtime::VarLen32 name, runtime::VarLen32 meta) {
   std::cout << "create table:" << name.str() << ", " << meta.str() << std::endl;
   if (hasTable(name.str())) {
      throw std::runtime_error("table " + name.str() + " does already exist");
   }
   createTable(name.str(), runtime::TableMetaData::deserialize(meta.str()));
}
void Database::copyFromIntoTable(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape) {
   arrow::io::IOContext ioContext = arrow::io::default_io_context();
   auto inputFile = arrow::io::ReadableFile::Open(fileName.str()).ValueOrDie();
   std::shared_ptr<arrow::io::InputStream> input = inputFile;

   auto readOptions = arrow::csv::ReadOptions::Defaults();

   auto parseOptions = arrow::csv::ParseOptions::Defaults();
   parseOptions.delimiter = delimiter.str().front();
   if (escape.getLen() > 0) {
      parseOptions.escape_char = escape.str().front();
      parseOptions.escaping = true;
   }
   parseOptions.newlines_in_values = true;
   auto convertOptions = arrow::csv::ConvertOptions::Defaults();
   auto schema = getTable(tableName)->schema();
   convertOptions.null_values.push_back("");
   convertOptions.strings_can_be_null = true;
   for (auto f : schema->fields()) {
      readOptions.column_names.push_back(f->name());
      convertOptions.column_types.insert({f->name(), f->type()});
   }

   // Instantiate TableReader from input stream and options
   auto maybeReader = arrow::csv::TableReader::Make(ioContext,
                                                    input,
                                                    readOptions,
                                                    parseOptions,
                                                    convertOptions);
   if (!maybeReader.ok()) {
      // Handle TableReader instantiation error...
   }
   std::shared_ptr<arrow::csv::TableReader> reader = *maybeReader;

   // Read table from CSV file
   auto maybeTable = reader->Read();
   if (!maybeTable.ok()) {
      // Handle CSV read error
      // (for example a CSV syntax error or failed type conversion)
   }
   std::shared_ptr<arrow::Table> table = *maybeTable;
   appendTable(tableName.str(), table);
}
void Database::setPersistMode(bool persist) {
   if (persist) {
      throw std::runtime_error("DB does not support persistent data");
   }
}
void Database::setPersist(bool persist) {
   setPersistMode(persist);
}
void Database::appendTable(runtime::VarLen32 tableName, ArrowTable* newRows) {
   appendTable(tableName.str(), *reinterpret_cast<std::shared_ptr<arrow::Table>*>(newRows));
}
} //end namespace runtime