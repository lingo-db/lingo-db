#include "bridge.h"
#include "execution/Execution.h"

#include "runtime/ArrowDirDatabase.h"
#include "runtime/ExternalArrowDatabase.h"
#include <iostream>
#include <arrow/pretty_print.h>
runtime::ExecutionContext* currentExecutionContext = nullptr;
void bridge::createInMemory() {
   currentExecutionContext = new runtime::ExecutionContext;
   currentExecutionContext->db = std::make_unique<runtime::ExternalArrowDatabase>();
}
void bridge::loadFromDisk(const char* directory) {
   currentExecutionContext = new runtime::ExecutionContext;
   auto database = runtime::Database::loadMetaDataAndSamplesFromDir(directory);
   currentExecutionContext->db = std::move(database);
}
void bridge::run(const char* module) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, false);
   queryExecutionConfig->resultProcessor = execution::createTablePrinter();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(module);
   executer->setExecutionContext(currentExecutionContext);
   executer->execute();
}
void bridge::runSQL(const char* query) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, true);
   queryExecutionConfig->resultProcessor = execution::createTablePrinter();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(query);
   executer->setExecutionContext(currentExecutionContext);
   executer->execute();
}
void bridge::addTable(const char* name, ArrowArrayStream* recordBatchStream) {
   auto recordBatchReader = arrow::ImportRecordBatchReader(recordBatchStream).ValueOrDie();
   currentExecutionContext->db->addTable(name, recordBatchReader->ToTable().ValueOrDie());
   arrow::PrettyPrint(*currentExecutionContext->db->getTable(name), {}, &std::cout);
}