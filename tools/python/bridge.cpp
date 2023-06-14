#include "bridge.h"
#include "execution/Execution.h"
#include "runtime/ArrowDirDatabase.h"
#include "runtime/ExternalArrowDatabase.h"
#include <arrow/table.h>
namespace bridge {
class Connection {
   runtime::ExecutionContext executionContext;

   public:
   Connection() : executionContext() {}
   void setDatabase(std::unique_ptr<runtime::Database> db) {
      executionContext.db = std::move(db);
   }
   runtime::ExecutionContext& getExecutionContext() {
      return executionContext;
   }
   runtime::Database& getDatabase() {
      return *executionContext.db;
   }
};
} //namespace bridge
bridge::Connection* bridge::createInMemory() {
   auto* res = new Connection;
   res->setDatabase(std::make_unique<runtime::ExternalArrowDatabase>());
   return res;
}
bridge::Connection* bridge::loadFromDisk(const char* directory) {
   auto* res = new Connection;
   res->setDatabase(runtime::Database::loadMetaDataAndSamplesFromDir(directory));
   return res;
}
bool bridge::run(Connection* connection, const char* module, ArrowArrayStream* res) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, false);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(module);
   executer->setExecutionContext(&connection->getExecutionContext());
   executer->execute();
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*result);
      if (!arrow::ExportRecordBatchReader(batchReader, res).ok()) {
         std::cerr << "export failed" << std::endl;
      }else{
         return true;
      }
   }
   return false;
}
bool bridge::runSQL(Connection* connection, const char* query, ArrowArrayStream* res) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, true);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(query);
   executer->setExecutionContext(&connection->getExecutionContext());
   executer->execute();
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*result);
      if (!arrow::ExportRecordBatchReader(batchReader, res).ok()) {
         std::cerr << "export failed" << std::endl;
      }else{
         return true;
      }
   }
   return false;
}
void bridge::addTable(Connection* connection, const char* name, ArrowArrayStream* recordBatchStream) {
   auto recordBatchReader = arrow::ImportRecordBatchReader(recordBatchStream).ValueOrDie();
   connection->getDatabase().addTable(name, recordBatchReader->ToTable().ValueOrDie());
}
void bridge::closeConnection(bridge::Connection* con) {
   delete con;
}