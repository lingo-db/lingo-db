#ifndef RUNTIME_ARROWDIRDATABASE_H
#define RUNTIME_ARROWDIRDATABASE_H
#include "runtime/Database.h"
namespace runtime {
class ArrowDirDatabase : public runtime::Database {
   std::string directory;
   bool writeback = true;
   std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tables;
   std::unordered_map<std::string, std::shared_ptr<arrow::RecordBatch>> samples;
   std::unordered_map<std::string, std::shared_ptr<TableMetaData>> metaData;
   static std::shared_ptr<arrow::RecordBatch> loadSample(std::string name);
   static std::shared_ptr<arrow::Table> loadTable(std::string name);
   void writeMetaData(std::string filename);

   public:
   std::shared_ptr<arrow::RecordBatch> getSample(const std::string& name) override;
   std::shared_ptr<arrow::Table> getTable(const std::string& name) override;
   std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name) override;
   bool hasTable(const std::string& name) override;
   static std::unique_ptr<Database> load(std::string dir);
   static std::unique_ptr<Database> empty() { return std::make_unique<ArrowDirDatabase>(); }
   void createTable(std::string tableName, std::shared_ptr<TableMetaData>) override;
   void appendTable(std::string tableName, std::shared_ptr<arrow::Table> newRows) override;
   void setWriteback(bool writeback);
   void setPersistMode(bool persist) override {
      setWriteback(persist);
   }
   ~ArrowDirDatabase();
};
} // end namespace runtime
#endif // RUNTIME_ARROWDIRDATABASE_H
