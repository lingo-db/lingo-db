#include "bridge.h"
#include "execution/Execution.h"
#include <arrow/table.h>
namespace bridge {
class Connection {
   std::shared_ptr<runtime::Session> session;

   public:
   Connection(std::shared_ptr<runtime::Session> session) : session(session) {}
   runtime::Session& getSession() {
      return *session;
   }
};
} //namespace bridge
bridge::Connection* bridge::createInMemory() {
   return new Connection(runtime::Session::createSession());
}
bridge::Connection* bridge::loadFromDisk(const char* directory) {
   return new Connection(runtime::Session::createSession(directory, true));
}
bool bridge::run(Connection* connection, const char* module, ArrowArrayStream* res) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, false);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), connection->getSession());
   executer->fromData(module);
   executer->execute();
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*result);
      if (!arrow::ExportRecordBatchReader(batchReader, res).ok()) {
         std::cerr << "export failed" << std::endl;
      } else {
         return true;
      }
   }
   return false;
}
bool bridge::runSQL(Connection* connection, const char* query, ArrowArrayStream* res) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, true);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), connection->getSession());
   executer->fromData(query);
   executer->execute();
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*result);
      if (!arrow::ExportRecordBatchReader(batchReader, res).ok()) {
         std::cerr << "export failed" << std::endl;
      } else {
         return true;
      }
   }
   return false;
}
void bridge::createTable(Connection* connection, const char* name, const char* metaData) {
   connection->getSession().getCatalog()->addTable(name, runtime::TableMetaData::create(metaData, name, {}));
}
void bridge::appendTable(Connection* connection, const char* name, ArrowArrayStream* recordBatchStream) {
   auto recordBatchReader = arrow::ImportRecordBatchReader(recordBatchStream).ValueOrDie();
   connection->getSession().getCatalog()->findRelation(name)->append(recordBatchReader->ToTable().ValueOrDie());
}
void bridge::closeConnection(bridge::Connection* con) {
   delete con;
}