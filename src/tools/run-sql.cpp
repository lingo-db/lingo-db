#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include <fstream>
#include <iostream>
#include <string>

namespace {
utility::GlobalSetting<bool> eagerLoading("system.eager_loading", false);
} // namespace
int main(int argc, char** argv) {
   using namespace lingodb;
   if (argc <= 2) {
      std::cerr << "USAGE: run-sql *.sql database" << std::endl;
      return 1;
   }
   std::string inputFileName = std::string(argv[1]);
   std::string directory = std::string(argv[2]);
   std::cout << "Loading Database from: " << directory << '\n';
   auto session = runtime::Session::createSession(directory, eagerLoading.getValue());

   lingodb::compiler::support::eval::init();
   execution::ExecutionMode runMode = execution::getExecutionMode();
   auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, true);
   if (const char* numRuns = std::getenv("QUERY_RUNS")) {
      queryExecutionConfig->executionBackend->setNumRepetitions(std::atoi(numRuns));
      std::cout << "using " << queryExecutionConfig->executionBackend->getNumRepetitions() << " runs" << std::endl;
   }
   unsetenv("PERF_BUILDID_DIR");
   queryExecutionConfig->timingProcessor = std::make_unique<execution::TimingPrinter>(inputFileName);

   auto scheduler = scheduler::startScheduler();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
   return 0;
}
