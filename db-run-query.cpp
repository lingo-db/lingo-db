#include <iostream>
#include <string>

#include "arrow/array.h"
#include "mlir-support/eval.h"
#include "runner/runner.h"


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

   runner::Runner runner;
   runner.load(inputFileName);
   //runner.dump();
   runner.optimize(*context.db);
   //runner.dump();
   runner.lower();
   //runner.dump();
   runner.lowerToLLVM();
   //runner.dump();
   //runner.dumpLLVM();
   runner.runJit(&context, 5, runner::Runner::printTable);
   return 0;
}