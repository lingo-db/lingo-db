#ifndef RUNTIME_DATABASE_H
#define RUNTIME_DATABASE_H

#include <memory>
#include <unordered_map>

#include "helpers.h"
#include "metadata.h"
#include <arrow/type_fwd.h>
namespace runtime {
struct ArrowTable {};
class ExecutionContext;
struct ExternalHashIndexMapping;
class Database {
   std::string directory;

   public:
   virtual void addTable(std::string tableName, std::shared_ptr<arrow::Table> table) = 0;
   static std::unique_ptr<Database> loadFromDir(std::string directory);
   virtual bool recordBatchesAvailable() {
      return false;
   }
   virtual const std::vector<std::shared_ptr<arrow::RecordBatch>>& getRecordBatches(const std::string& name);
   static std::unique_ptr<Database> loadMetaDataAndSamplesFromDir(std::string directory);
   static std::shared_ptr<arrow::Table> loadTable(std::string name);

   virtual bool hasTable(const std::string& name) = 0;
   virtual bool hasTableInMetadata(const std::string& tableName) = 0;

   virtual std::shared_ptr<arrow::Table> getTable(const std::string& name) = 0;
   virtual ExternalHashIndexMapping* getIndex(const std::string& name, const std::vector<std::string>& mapping);
   virtual void addIndex(std::string& tableName);
   virtual std::shared_ptr<arrow::RecordBatch> getSample(const std::string& name) = 0;
   virtual std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name) = 0;
   virtual void createTable(std::string tableName, std::shared_ptr<TableMetaData>);
   virtual void appendTable(std::string tableName, std::shared_ptr<arrow::Table> newRows);
   virtual void combineTableWithHashValuesImpl(std::string tableName, std::shared_ptr<arrow::Table> hashValues);

   void createTable(runtime::VarLen32 name, runtime::VarLen32 meta);
   void appendTableFromResult(runtime::VarLen32 tableName, runtime::ExecutionContext* context, size_t resultId);
   void combineTableWithHashValues(runtime::VarLen32 tableName, runtime::ExecutionContext* context, size_t resultId);

   virtual void setPersistMode(bool persist);
   void setPersist(bool persist);
   void setDirectory(std::string directory);
   std::string getDirectory();
   void copyFromIntoTable(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape);
   static std::string serializeRecordBatch(std::shared_ptr<arrow::RecordBatch> batch);
   static std::shared_ptr<arrow::RecordBatch> deserializeRecordBatch(std::string str);
   virtual ~Database() {}
};

} //end namespace runtime

#endif // RUNTIME_DATABASE_H
