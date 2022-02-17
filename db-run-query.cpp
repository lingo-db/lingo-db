#include <fstream>
#include <iostream>
#include <string>

#include "arrow/array.h"
#include "mlir-support/eval.h"
#include "runner/runner.h"

bool beingTraced() {
   std::ifstream sf("/proc/self/status");
   std::string s;
   while (sf >> s) {
      if (s == "TracerPid:") {
         int pid;
         sf >> pid;
         return pid != 0;
      }
      std::getline(sf, s);
   }

   return false;
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
      auto database = runtime::Database::load(std::string(argv[2]));
      context.db = std::move(database);
   }
   support::eval::init();
   runner::RunMode runMode;
   if (RUN_QUERIES_WITH_PERF) {
      runMode = runner::RunMode::PERF;
   } else {
      runMode = beingTraced() ? runner::RunMode::DEBUGGING : runner::RunMode::SPEED;
   }
   runner::Runner runner(runMode);
   runner.load(inputFileName);
   runner.optimize(*context.db);
   runner.lower();
   //runner.dump();
   runner.lowerToLLVM();
   runner.runJit(&context, 5, runner::Runner::printTable);
   return 0;
}