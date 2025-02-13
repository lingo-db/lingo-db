#include <fstream>
#include <iostream>
#include <string>

#include "lingodb/execution/Execution.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/scheduler/Scheduler.h"

int main(int argc, char** argv) {
   using namespace lingodb;
   std::string inputFileName = "-";
   if (argc > 1) {
      inputFileName = std::string(argv[1]);
   }

   bool eagerLoading = std::getenv("LINGODB_BACKEND_ONLY");
   std::shared_ptr<runtime::Session> session;
   if (argc > 2) {
      std::cout << "Loading Database from: " << argv[2] << '\n';
      session = runtime::Session::createSession(std::string(argv[2]), eagerLoading);
   } else {
      session = runtime::Session::createSession();
   }
   lingodb::compiler::support::eval::init();
   execution::ExecutionMode runMode = execution::getExecutionMode();
   auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, false);
   if (const char* numRuns = std::getenv("QUERY_RUNS")) {
      queryExecutionConfig->executionBackend->setNumRepetitions(std::atoi(numRuns));
      std::cout << "using " << queryExecutionConfig->executionBackend->getNumRepetitions() << " runs" << std::endl;
   }
   if (std::getenv("LINGODB_BACKEND_ONLY")) {
      queryExecutionConfig->queryOptimizer = {};
      queryExecutionConfig->loweringSteps.clear();
   }
   queryExecutionConfig->timingProcessor = std::make_unique<execution::TimingPrinter>(inputFileName);

   auto scheduler = scheduler::startScheduler();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
   return 0;
}
