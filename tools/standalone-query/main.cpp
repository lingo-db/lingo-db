#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"
#include <fstream>
#include <iostream>
#include <string>

#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/scheduler/Tasks.h"
extern "C" void mainFunc();
namespace {

class RunCompiledQueryTask : public lingodb::scheduler::TaskWithContext {
   lingodb::runtime::ExecutionContext& context;

   public:
   RunCompiledQueryTask(lingodb::runtime::ExecutionContext& context) : TaskWithContext(&context), context(context) {}
   bool allocateWork() override {
      if (workExhausted.exchange(true)) {
         return false;
      }
      return true;
   }
   void performWork() override {
      auto start = std::chrono::high_resolution_clock::now();
      mainFunc();
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "Execution took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;
      auto tablePrinter = lingodb::execution::createTablePrinter();
      tablePrinter->process(&context);
   }
};
} // namespace
int main(int argc, char** argv) {
   using namespace lingodb;
   if (argc <= 1) {
      std::cerr << "USAGE: run-sql database" << std::endl;
      return 1;
   }
   std::string directory = std::string(argv[1]);
   std::cout << "Loading Database from: " << directory << '\n';
   auto session = runtime::Session::createSession(directory, true);

   auto scheduler = scheduler::startScheduler();
   auto context = session->createExecutionContext();
   scheduler::awaitEntryTask(std::make_unique<RunCompiledQueryTask>(*context));
   return 0;
}
