#include "runtime/ExternalArrowDatabase.h"

#include <iostream>
using namespace runtime;
void runtime::ExternalArrowDatabase::addTable(std::string name, std::shared_ptr<arrow::Table> table) {
   std::cout << "adding table:" << name << std::endl;
   tables[name] = table;
}
std::shared_ptr<arrow::Table> runtime::ExternalArrowDatabase::getTable(const std::string& name) {
   if (tables.contains(name)) {
      return tables.at(name);
   } else {
      throw std::runtime_error("unknown table:" + name);
   }
   return std::shared_ptr<arrow::Table>();
}
std::shared_ptr<arrow::RecordBatch> runtime::ExternalArrowDatabase::getSample(const std::string& name) {
   return std::shared_ptr<arrow::RecordBatch>();
}
std::shared_ptr<TableMetaData> runtime::ExternalArrowDatabase::getTableMetaData(const std::string& name) {
   return runtime::TableMetaData::create("", name, {});
}
bool ExternalArrowDatabase::hasTable(const std::string& name) {
   return tables.contains(name);
}
