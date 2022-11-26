#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "arrow/array.h"
#include "execution/runner.h"
#include "mlir-support/eval.h"
void check(bool b, std::string message) {
   if (!b) {
      std::cerr << "ERROR: " << message << std::endl;
      exit(1);
   }
}
void handleQuery(runtime::ExecutionContext* context, std::string basicString);

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
      if (line == "exit" || std::cin.eof()) {
         //exit from repl loop
         break;
      }
      while (std::cin.good()) {
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
   auto queryExecutionConfig = runner::createQueryExecutionConfig(runner::RunMode::DEFAULT, true);
   auto executer = runner::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(sqlQuery);
   executer->setExecutionContext(context);
   executer->execute();
}
