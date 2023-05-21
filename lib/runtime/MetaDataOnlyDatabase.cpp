#include "runtime/MetaDataOnlyDatabase.h"

#include "json.h"
#include <fstream>
using namespace runtime;
std::unique_ptr<runtime::Database> MetaDataOnlyDatabase::loadMetaData(std::string file) {
   auto database = std::make_unique<MetaDataOnlyDatabase>();
   std::ifstream t(file);

   auto metadataString = std::string((std::istreambuf_iterator<char>(t)),
                                     std::istreambuf_iterator<char>());
   auto metaDataJSON = nlohmann::json::parse(metadataString);
   for (auto& table : metaDataJSON["tables"].items()) {
      database->metaData[table.key()] = runtime::TableMetaData::create(metadataString, table.key(), std::shared_ptr<arrow::RecordBatch>());
   }
   return database;
}
std::unique_ptr<runtime::Database> MetaDataOnlyDatabase::emptyMetaData() {
   return std::make_unique<MetaDataOnlyDatabase>();
}
bool MetaDataOnlyDatabase::hasTable(const std::string& name) {
   return metaData.contains(name);
}
bool MetaDataOnlyDatabase::hasTableInMetadata(const std::string& tableName){
   return hasTable(tableName);
}
void MetaDataOnlyDatabase::addTable(std::string tableName, std::shared_ptr<arrow::Table> table) {
   throw std::runtime_error("addTable on MetaDataOnlyDatabase");
}
std::shared_ptr<arrow::Table> MetaDataOnlyDatabase::getTable(const std::string& name) {
   throw std::runtime_error("getTable on MetaDataOnlyDatabase");
}
std::shared_ptr<arrow::RecordBatch> MetaDataOnlyDatabase::getSample(const std::string& name) {
   throw std::runtime_error("getSample on MetaDataOnlyDatabase");
}
std::shared_ptr<TableMetaData> MetaDataOnlyDatabase::getTableMetaData(const std::string& name) {
   return metaData.at(name);
}
