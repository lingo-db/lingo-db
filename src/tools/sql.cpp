#include "features.h"
#include "linenoise.h"

#include <fstream>
#include <iostream>
#include <string>

#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/utility/Setting.h"

namespace {
using namespace lingodb;
utility::GlobalSetting<bool> reportTimes("sql.report_times", false);
utility::GlobalSetting<std::string> reportFormat("sql.report_format", "simple");
utility::GlobalSetting<bool> prompt("sql.prompt", true);

class ConciseConsoleTimingPrinter : public execution::TimingProcessor {
   double compilation;
   double execution;

   public:
   ConciseConsoleTimingPrinter() : compilation(0.0), execution(0.0) {
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

class LoggingTimingPrinter : public execution::TimingProcessor {
   std::unordered_map<std::string, double> timings;

   public:
   LoggingTimingPrinter() = default;

   void addTiming(const std::unordered_map<std::string, double>& timing) override {
      for (auto [name, t] : timing) {
         if (name == "executionTime") {
            timings["execution"] += t;
         } else if (name == "QOpt") {
            timings["qopt"] += t;
         } else if (name == "lowerRelAlg") {
            timings["lowerrelalg"] += t;
         } else if (name == "lowerSubOp") {
            timings["lowersubop"] += t;
         } else if (name == "lowerDB") {
            timings["lowerdb"] += t;
         } else if (name == "lowerArrow") {
            timings["lowerarrow"] += t;
         } else if (name == "lowerToLLVM" || name == "baselineLowering" || name == "toLLVMIR") {
            timings["lowerforbackend"] += t;
         } else if (name == "llvmOptimize") {
            timings["backendoptimize"] += t;
         } else if (name == "llvmCodeGen" || name == "baselineCodeGen") {
            timings["backendcodegen"] += t;
         } else if (name == "baselineEmit") {
            timings["backendlink"] += t;
         } else {
            std::cerr << "Unknown timing entry: " << name << " with value: " << t << std::endl;
            abort();
         }
      }
   }

   void process() override {
      // Output CSV header
      for (auto it = timings.begin(); it != timings.end(); ++it) {
         std::cerr << it->first;
         if (std::next(it) != timings.end()) std::cerr << ",";
      }
      std::cerr << std::endl;
      // Output CSV values
      for (auto it = timings.begin(); it != timings.end(); ++it) {
         std::cerr << it->second;
         if (std::next(it) != timings.end()) std::cerr << ",";
      }
      std::cerr << std::endl;
   }
};
void handleQuery(runtime::Session& session, std::string sqlQuery) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::getExecutionMode(), true);
   if (reportTimes.getValue()) {
      if (reportFormat.getValue() == "simple") {
         queryExecutionConfig->timingProcessor = std::make_unique<ConciseConsoleTimingPrinter>();
      } else if (reportFormat.getValue() == "csv") {
         queryExecutionConfig->timingProcessor = std::make_unique<LoggingTimingPrinter>();
      } else {
         std::cerr << "Unknown report format: " << reportFormat.getValue() << std::endl;
         return;
      }
   }
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), session);
   executer->fromData(sqlQuery);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
}
} // namespace
int main(int argc, char** argv) {
   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }

   if (argc <= 1) {
      std::cerr << "USAGE: sql database" << std::endl;
      return 1;
   }
   auto session = runtime::Session::createSession(std::string(argv[1]), true);

   compiler::support::eval::init();
   auto scheduler = scheduler::startScheduler();

   linenoiseSetMultiLine(true); // enables multi-line editing
   std::stringstream query;
   size_t count = 0;
   while (true) {
      const char* promptStr = prompt.getValue() ? (query.str().empty() ? "sql> " : "   -> ") : "";
      char* input = linenoise(promptStr);

      if (input == nullptr) {
         // Ctrl+D or EOF
         std::cout << std::endl;
         break;
      }

      std::string line = input;
      free(input); // linenoise returns malloc'd memory

      if (line == "exit") {
         break;
      }

      query << line << '\n';

      // Trim trailing whitespace
      std::string trimmed = line;
      trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
      count+= std::ranges::count(trimmed.begin(), trimmed.end(), '$');

      if (!trimmed.empty() && trimmed.back() == ';' && count%4 == 0) {
         // Done reading the full statement
         linenoiseHistoryAdd(query.str().c_str()); // optional: add to history

         handleQuery(*session, query.str());

         // Clear for next command
         query.str("");
         query.clear();
      }
   }
   linenoiseHistoryFree();

   return 0;
}
