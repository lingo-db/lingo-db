#ifndef RUNTIME_DATABASE_H
#define RUNTIME_DATABASE_H

#include <memory>
#include <unordered_map>

#include "helpers.h"
#include "metadata.h"
#include <arrow/type_fwd.h>
namespace runtime {
struct ArrowTable{};
class Database {
   public:
   //void addTable(std::string name, std::shared_ptr<arrow::Table> table);
   static std::unique_ptr<Database> loadFromDir(std::string directory);
   virtual bool hasTable(const std::string& name) = 0;

   virtual std::shared_ptr<arrow::Table> getTable(const std::string& name) = 0;
   virtual std::shared_ptr<arrow::RecordBatch> getSample(const std::string& name) = 0;
   virtual std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name) = 0;
   virtual void createTable(std::string tableName, std::shared_ptr<TableMetaData>);
   virtual void appendTable(std::string tableName, std::shared_ptr<arrow::Table> newRows);
   void createTable(runtime::VarLen32 name, runtime::VarLen32 meta);
   void appendTable(runtime::VarLen32 tableName, ArrowTable* newRows);

   virtual void setPersistMode(bool persist);
   void setPersist(bool persist);
   void copyFromIntoTable(runtime::VarLen32 tableName,runtime::VarLen32 fileName,runtime::VarLen32 delimiter,runtime::VarLen32 escape);
   static std::string serializeRecordBatch(std::shared_ptr<arrow::RecordBatch> batch);
   static std::shared_ptr<arrow::RecordBatch> deserializeRecordBatch(std::string str);
   virtual ~Database() {}
};

} //end namespace runtime

#endif // RUNTIME_DATABASE_H
