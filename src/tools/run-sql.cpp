#include "features.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/runtime/SIP.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include <fstream>
#include <iostream>
#include <string>

namespace {
lingodb::utility::GlobalSetting<bool> eagerLoading("system.eager_loading", false);
} // namespace
int main(int argc, char** argv) {
   using namespace lingodb;

   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }

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
   unsetenv("PERF_BUILDID_DIR");
   queryExecutionConfig->timingProcessor = std::make_unique<execution::TimingPrinter>(inputFileName);

   auto scheduler = scheduler::startScheduler();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
#if DEBUG

   auto* sipNode = lingodb::runtime::SIP::sips.load();
   if (sipNode) {
      std::cerr << "SIP STATS------------------\n";
   }
   while (sipNode != nullptr) {
      std::cerr << "SIP " << std::to_string(sipNode->id) << " stats:\n";
      std::cerr << "  complete count: " << sipNode->completeCount.load() << "\n";
      std::cerr << "  filtered count: " << sipNode->filteredCount.load() << "\n";
      if (sipNode->skipState == 2) {
         std::cerr << "  Skipped" << std::endl;
      }
      std::cerr << "  Perc: " << (100.0 * sipNode->filteredCount.load() / sipNode->completeCount.load()) << "%\n";

      sipNode = sipNode->next;
   }

#endif

   return 0;
}
