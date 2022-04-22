#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "arrow/array.h"
#include "mlir-support/eval.h"
#include "runner/runner.h"

void handleQuery(runtime::ExecutionContext* Context, std::string basicString);
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
   runtime::ExecutionContext context;
   context.id = 42;
   if (argc <= 1) {
      std::cerr << "USAGE: sql database" << std::endl;
      return 1;
   }
   std::cout << "Loading Database from: " << argv[1] << '\n';
   auto database = runtime::Database::loadFromDir(std::string(argv[1]));
   context.db = std::move(database);

   support::eval::init();
   while (true) {
      //print prompt
      std::cout << "sql>";
      //read query from stdin until semicolon appears
      std::stringstream query;
      std::string line;
      std::getline(std::cin, line);
      if (line == "exit") {
         //exit from repl loop
         break;
      }
      while (true) {
         query << line << std::endl;
         if (!line.empty() && line.find(';') == line.size() - 1) {
            break;
         }
         std::getline(std::cin, line);
      }
      handleQuery(&context, query.str());
   }

   return 0;
}
void handleQuery(runtime::ExecutionContext* context, std::string sqlQuery) {
   runner::Runner runner(runner::RunMode::SPEED);
   runner.loadSQL(sqlQuery, *context->db);
   runner.dump();
   runner.optimize(*context->db);
   //runner.dump();
   runner.lower();
   //runner.dump();
   runner.lowerToLLVM();
   size_t runs = 1;
   runner.runJit(context, runs, runner::Runner::printTable);
}
