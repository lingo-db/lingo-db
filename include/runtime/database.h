#ifndef RUNTIME_DATABASE_H
#define RUNTIME_DATABASE_H

#include <arrow/table.h>

#include <memory>
#include <unordered_map>
namespace runtime {

class Database {
   std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tables;
   static std::shared_ptr<arrow::Table> loadTable(std::string name);

   public:
   static std::unique_ptr<Database> load(std::string directory);
   std::shared_ptr<arrow::Table> getTable(const std::string& name);
};

} //end namespace runtime

#endif // RUNTIME_DATABASE_H
