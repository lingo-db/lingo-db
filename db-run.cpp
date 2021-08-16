#include "arrow/array.h"
#include "runner/runner.h"
#include <iostream>
#include <string>

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

   runner::Runner runner;
   runner.load(inputFileName);
   runner.lower();
   //runner.dump();
   runner.lowerToLLVM();
   //runner.dumpLLVM();
   runner.runJit(&context, runner::Runner::printTable);
   return 0;
}