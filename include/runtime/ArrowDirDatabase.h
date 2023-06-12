#ifndef RUNTIME_ARROWDIRDATABASE_H
#define RUNTIME_ARROWDIRDATABASE_H
#include "ExternalHashIndex.h"
#include "runtime/Database.h"
namespace runtime {
class ArrowDirDatabase : public runtime::Database {
   bool writeback = true;
   std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tables;
   std::unordered_map<std::string, std::vector<std::shared_ptr<arrow::RecordBatch>>> recordBatches;
   std::unordered_map<std::string, std::shared_ptr<arrow::RecordBatch>> samples;
   std::unordered_map<std::string, std::shared_ptr<TableMetaData>> metaData;
   ExternalHashIndexManager externalHashIndexManager;
   static std::shared_ptr<arrow::RecordBatch> loadSample(std::string name);
   void writeMetaData(std::string filename);
   void updateRecordBatches(const std::string& name);
   public:
   std::shared_ptr<arrow::RecordBatch> getSample(const std::string& name) override;
   std::shared_ptr<arrow::Table> getTable(const std::string& name) override;
   bool recordBatchesAvailable() override { return true; }
   const std::vector<std::shared_ptr<arrow::RecordBatch>> & getRecordBatches(const std::string &name) override{
      return recordBatches.at(name);
   }
   std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name) override;
   ExternalHashIndexMapping* getIndex(const std::string& name, const std::vector<std::string>& mapping) override;
   void addIndex(std::string& name) override;
   bool hasTable(const std::string& name) override;
   bool hasTableInMetadata(const std::string& tableName) override;
   static std::unique_ptr<Database> empty() { return std::make_unique<ArrowDirDatabase>(); }
   static std::unique_ptr<Database> load(std::string dir, bool loadTables = true);
   static std::shared_ptr<arrow::Table> loadTable(std::string name);
   void addTable(std::string tableName, std::shared_ptr<arrow::Table> table) override;
   void createTable_(std::string tableName, std::shared_ptr<TableMetaData>) override;
   void appendTable(std::string tableName, std::shared_ptr<arrow::Table> newRows) override;
   void combineTableWithHashValuesImpl(std::string tableName, std::shared_ptr<arrow::Table> hashValues) override;

   void setWriteback(bool writeback);
   void setPersistMode(bool persist) override {
      setWriteback(persist);
   }
   ~ArrowDirDatabase();
};
} // end namespace runtime
#endif // RUNTIME_ARROWDIRDATABASE_H
