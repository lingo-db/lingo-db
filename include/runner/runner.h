#ifndef DB_DIALECTS_RUNNER_H
#define DB_DIALECTS_RUNNER_H

#include <string>
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
#endif //DB_DIALECTS_RUNNER_H
