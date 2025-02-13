#include <fstream>
#include <iostream>
#include <string>

#include "lingodb/execution/Execution.h"
#include "lingodb/compiler/mlir-support/eval.h"

namespace {
using namespace lingodb;
class ConciseTimingPrinter : public execution::TimingProcessor {
   double compilation;
   double execution;

   public:
   ConciseTimingPrinter() : compilation(0.0), execution(0.0) {
   }
   void addTiming(const std::unordered_map<std::string, double>& timing) override {
      for (auto [name, t] : timing) {
         if (name == "executionTime") {
            execution = t;
         } else {
            compilation += t;
         }
      }
   }
   void process() override {
      std::cerr << " compilation: " << compilation << " [ms] execution: " << execution << " [ms]" << std::endl;
   }
};
void handleQuery(runtime::Session& session, std::string sqlQuery, bool reportTimes) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::DEFAULT, true);
   if (reportTimes) {
      queryExecutionConfig->timingProcessor = std::make_unique<ConciseTimingPrinter>();
   }
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), session);
   executer->fromData(sqlQuery);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
}
} // namespace
int main(int argc, char** argv) {
   if (argc <= 1) {
      std::cerr << "USAGE: sql database" << std::endl;
      return 1;
   }
   bool reportTimes=false;
   if(const char* reportTimesEnv=std::getenv("LINGODB_SQL_REPORT_TIMES")){
      reportTimes= std::stoll(reportTimesEnv);
   }
   bool prompt=true;
   if(const char* promptEnv=std::getenv("LINGODB_SQL_PROMPT")){
      prompt= std::stoll(promptEnv);
   }

   auto session = runtime::Session::createSession(std::string(argv[1]), true);

   lingodb::compiler::support::eval::init();
   auto scheduler = scheduler::startScheduler();
   while (true) {
      //print prompt
      if (prompt){
         std::cout << "sql>";
      }
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
      handleQuery(*session, query.str(), reportTimes);
   }

   return 0;
}
