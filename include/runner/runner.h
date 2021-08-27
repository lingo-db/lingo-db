#ifndef RUNNER_RUNNER_H
#define RUNNER_RUNNER_H

#include <functional>
#include <string>

#include "runtime/execution_context.h"

namespace runner {
class Runner {
   public:
   Runner();
   bool load(std::string fileName);
   bool optimize();
   bool lower();
   bool lowerToLLVM();
   void dump();
   bool runJit(runtime::ExecutionContext* context, std::function<void(uint8_t*)> callback);
   ~Runner();
   static void printTable(uint8_t* ptr);

   private:
   void* context;
};
} // namespace runner
#endif // RUNNER_RUNNER_H
