#include "linenoise.h"

#include <fstream>
#include <iostream>
#include <string>

#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"

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
   auto queryExecutionConfig = execution::createQueryExecutionConfigWithNewFrontend(execution::getExecutionMode(), true);
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
   bool reportTimes = false;
   if (const char* reportTimesEnv = std::getenv("LINGODB_SQL_REPORT_TIMES")) {
      reportTimes = std::stoll(reportTimesEnv);
   }
   bool prompt = true;
   if (const char* promptEnv = std::getenv("LINGODB_SQL_PROMPT")) {
      prompt = std::stoll(promptEnv);
   }

   auto session = runtime::Session::createSession(std::string(argv[1]), true);

   lingodb::compiler::support::eval::init();
   auto scheduler = scheduler::startScheduler();

   linenoiseSetMultiLine(true); // enables multi-line editing
   std::string line;
   std::stringstream query;

   while (true) {
      const char* promptStr = prompt ? (query.str().empty() ? "sql> " : "   -> ") : "";
      char* input = linenoise(promptStr);

      if (input == nullptr) {
         // Ctrl+D or EOF
         std::cout << std::endl;
         break;
      }

      line = input;
      free(input); // linenoise returns malloc'd memory

      if (line == "exit") {
         break;
      }

      query << line << '\n';

      // Trim trailing whitespace
      std::string trimmed = line;
      trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);

      if (!trimmed.empty() && trimmed.back() == ';') {
         // Done reading the full statement
         linenoiseHistoryAdd(query.str().c_str()); // optional: add to history

         handleQuery(*session, query.str(), reportTimes);

         // Clear for next command
         query.str("");
         query.clear();
      }
   }
   linenoiseHistoryFree();

   return 0;
}
