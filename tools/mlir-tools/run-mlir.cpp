#include <fstream>
#include <iostream>
#include <string>

#include "execution/Execution.h"
#include "mlir-support/eval.h"

void check(bool b, std::string message) {
   if (!b) {
      std::cerr << "ERROR: " << message << std::endl;
      exit(1);
   }
}
int main(int argc, char** argv) {
   std::string inputFileName = "-";
   if (argc > 1) {
      inputFileName = std::string(argv[1]);
   }

   runtime::ExecutionContext context;
   context.id = 42;
   if (argc > 2) {
      std::cout << "Loading Database from: " << argv[2] << '\n';
      auto database = runtime::Database::loadFromDir(std::string(argv[2]));
      context.db = std::move(database);
   }
   support::eval::init();
   execution::ExecutionMode runMode = execution::getExecutionMode();
   auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, false);
   if (const char* numRuns = std::getenv("QUERY_RUNS")) {
      queryExecutionConfig->executionBackend->setNumRepetitions(std::atoi(numRuns));
      std::cout << "using " << queryExecutionConfig->executionBackend->getNumRepetitions() << " runs" << std::endl;
   }
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromFile(inputFileName);
   executer->setExecutionContext(&context);
   executer->execute();
   return 0;
}
