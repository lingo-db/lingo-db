#include <fstream>
#include <iostream>
#include <string>

#include "execution/Execution.h"
#include "mlir-support/eval.h"
void check(bool b, std::string message) {
   if (!b) {
      std::cerr << "ERROR: " << message << std::endl;
      exit(1);
   }
}
void handleQuery(runtime::Session& session, std::string sqlQuery) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::DEFAULT, true);
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), session);
   executer->fromData(sqlQuery);
   executer->execute();
}
int main(int argc, char** argv) {
   if (argc <= 1) {
      std::cerr << "USAGE: sql database" << std::endl;
      return 1;
   }
   auto session = runtime::Session::createSession(std::string(argv[1]),true);

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
      handleQuery(*session, query.str());
   }

   return 0;
}

