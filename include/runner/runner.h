#ifndef RUNNER_RUNNER_H
#define RUNNER_RUNNER_H

#include <functional>
#include <string>

#include "runtime/execution_context.h"

namespace runner {

enum class RunMode{
   SPEED=0, PERF=1, DEBUGGING=2
};
class Runner {
   public:
   Runner(RunMode runMode);
   bool loadSQL(std::string sql);
   bool load(std::string fileName);
   bool loadString(std::string input);
   bool optimize(runtime::Database& db);
   bool lower();
   bool lowerToLLVM();
   void dump();
   void snapshot();
   bool runJit(runtime::ExecutionContext* context,size_t repeats, std::function<void(uint8_t*)> callback);
   ~Runner();
   static void printTable(uint8_t* ptr);

   private:
   void* context;
   RunMode runMode;
};
} // namespace runner
#endif // RUNNER_RUNNER_H
