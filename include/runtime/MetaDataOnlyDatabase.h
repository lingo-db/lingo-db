
#ifndef RUNTIME_METADATAONLYDATABASE_H
#define RUNTIME_METADATAONLYDATABASE_H
#include "runtime/Database.h"
namespace runtime {
class MetaDataOnlyDatabase : public runtime::Database {
   std::unordered_map<std::string, std::shared_ptr<TableMetaData>> metaData;


   public:
   static std::unique_ptr<runtime::Database> loadMetaData(std::string file);
   static std::unique_ptr<runtime::Database> emptyMetaData();
   bool hasTable(const std::string& name) override;
   std::shared_ptr<arrow::Table> getTable(const std::string& name) override;
   std::shared_ptr<arrow::RecordBatch> getSample(const std::string& name) override;
   std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name) override;
};
} // end namespace runtime
#endif // RUNTIME_METADATAONLYDATABASE_H
