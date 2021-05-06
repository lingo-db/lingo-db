#include "runtime/database.h"
#include <filesystem>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/status.h>
#include <arrow/table.h>
namespace runtime {
std::shared_ptr<arrow::Table> Database::loadTable(std::string name) {
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   for (int i = 0; i < batchReader->num_record_batches(); i++) {
      batches.push_back(batchReader->ReadRecordBatch(i).ValueOrDie());
   }
   return arrow::Table::FromRecordBatches(batchReader->schema(), batches).ValueOrDie();
}

std::unique_ptr<Database> Database::load(std::string directory) {
   auto database = std::make_unique<Database>();
   for (const auto& p : std::filesystem::directory_iterator(directory)) {
      auto path = p.path();
      if (path.extension().string() == ".arrow") {
         auto tablename = path.stem().string();
         database->tables[tablename] = loadTable(path.string());
      }
   }
   return database;
}
std::shared_ptr<arrow::Table> Database::getTable(const std::string& name) {
   return tables[name];
}
} //end namespace runtime