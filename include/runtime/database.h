#ifndef RUNTIME_DATABASE_H
#define RUNTIME_DATABASE_H

#include <arrow/table.h>

#include <memory>
#include <unordered_map>

#include "metadata.h"
namespace runtime {

class Database {
   std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tables;
   std::unordered_map<std::string, std::shared_ptr<arrow::RecordBatch>> samples;
   std::unordered_map<std::string, std::shared_ptr<TableMetaData>> metaData;

   static std::shared_ptr<arrow::Table> loadTable(std::string name);
   static std::shared_ptr<arrow::RecordBatch> loadSample(std::string name);

   public:
   Database(){};

   void addTable(std::string name, std::shared_ptr<arrow::Table> table);
   static std::unique_ptr<Database> load(std::string directory);
   std::shared_ptr<arrow::Table> getTable(const std::string& name);
   std::shared_ptr<arrow::RecordBatch> getSample(const std::string& name);
   std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name);
   static std::string serializeRecordBatch(std::shared_ptr<arrow::RecordBatch> batch);
   static std::shared_ptr<arrow::RecordBatch> deserializeRecordBatch(std::string str);
};

} //end namespace runtime

#endif // RUNTIME_DATABASE_H
