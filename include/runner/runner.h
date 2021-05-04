#ifndef RUNNER_RUNNER_H
#define RUNNER_RUNNER_H

#include <string>
namespace runner {
class Runner {
   public:
   Runner();
   bool load(std::string fileName);
   bool lower();
   bool lowerToLLVM();
   void dump();
   void dumpLLVM();
   bool runJit();
   ~Runner();

   private:
   void* context;
};
}  // namespace runner
#endif // RUNNER_RUNNER_H
