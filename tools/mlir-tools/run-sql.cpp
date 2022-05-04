#include <fstream>
#include <iostream>
#include <sstream>
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
   std::string inputFileName = std::string(argv[1]);
   std::ifstream istream{inputFileName};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   std::string sqlQuery = buffer.str();
   std::cerr << sqlQuery << std::endl;
   runtime::ExecutionContext context;
   context.id = 42;
   if (argc <= 2) {
      std::cerr << "USAGE: run-sql *.sql database" << std::endl;
      return 1;
   }
   std::cout << "Loading Database from: " << argv[2] << '\n';
   auto database = runtime::Database::loadFromDir(std::string(argv[2]));
   context.db = std::move(database);

   support::eval::init();
   runner::RunMode runMode = runner::Runner::getRunMode();
   runner::Runner runner(runMode);
   check(runner.loadSQL(sqlQuery, *context.db), "SQL translation failed");
   check(runner.optimize(*context.db), "query optimization failed");
   check(runner.lower(), "could not lower DSA/DB dialects");
   check(runner.lowerToLLVM(), "lowering to llvm failed");
   size_t runs = 1;

   if (const char* numRuns = std::getenv("QUERY_RUNS")) {
      runs = std::atoi(numRuns);
      std::cout << "using " << runs << " runs" << std::endl;
   }
   check(runner.runJit(&context, runs, runner::Runner::printTable), "JIT execution failed");
   return 0;
}
