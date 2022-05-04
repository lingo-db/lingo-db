#include <fstream>
#include <iostream>
#include <string>

#include "arrow/array.h"
#include "mlir-support/eval.h"
#include "runner/runner.h"

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
   runner::RunMode runMode = runner::Runner::getRunMode();
   runner::Runner runner(runMode);
   check(runner.load(inputFileName), "could not load MLIR module");
   check(runner.optimize(*context.db), "query optimization failed");
   check(runner.lower(), "could not lower DSA/DB dialects");
   check(runner.lowerToLLVM(), "lowering to llvm failed");
   size_t runs = 1;
   if (const char* numRuns = std::getenv("QUERY_RUNS")) {
      runs = std::atoi(numRuns);
      std::cout << "using " << runs << " runs" << std::endl;
   }
   runner.runJit(&context, runs, runner::Runner::printTable);
   return 0;
}
