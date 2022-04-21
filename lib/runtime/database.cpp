#include <filesystem>
#include <fstream>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/status.h>

#include "runtime/ArrowDirDatabase.h"
#include "runtime/Database.h"

namespace runtime {

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

} //end namespace runtime