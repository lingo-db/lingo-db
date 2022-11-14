#ifndef RUNNER_RUNNER_H
#define RUNNER_RUNNER_H

#include <functional>
#include <string>

#include "runtime/ExecutionContext.h"

namespace runner {

enum class RunMode {
   SPEED = 0, //Aim for maximum speed (no verification of generated MLIR
   DEFAULT = 1, //Execute without introducing extra steps for debugging/profiling, but verify generated MLIR
   PERF = 2, //Profiling
   DEBUGGING = 3 //Make generated code debuggable
};
class Runner {
   public:
   Runner(RunMode runMode);
   bool loadSQL(std::string sql, runtime::Database& db);
   bool load(std::string fileName);
   bool loadString(std::string input);
   bool optimize(runtime::Database& db);
   bool lower();
   bool lowerToLLVM();
   void dump();
   void snapshot(std::string fileName="");
   bool runJit(runtime::ExecutionContext* context, size_t repeats, std::function<void(uint8_t*)> callback);
   ~Runner();
   static void printTable(uint8_t* ptr);
   enum SortMode {
      NONE,
      SORT,
      SORTROWS
   };
   static std::function<void(uint8_t*)> hashResult(SortMode sortMode, size_t& numValues, std::string& result, std::string& lines,bool tsv);
   static RunMode getRunMode();
   bool isReportTimes() const {
      return reportTimes;
   }
   void setReportTimes(bool reportTimes) {
      Runner::reportTimes = reportTimes;
   }

   private:
   void* context;
   RunMode runMode;
   bool reportTimes=true;
};
} // namespace runner
#endif // RUNNER_RUNNER_H
