#ifndef RUNTIME_TIMING_H
#define RUNTIME_TIMING_H
#include <cstdint>
namespace runtime {
class Timing {
   public:
   static uint64_t start();
   static void startPerf();
   static void stopPerf();
   static void stop(uint64_t start);
};
} // end namespace runtime
#endif //RUNTIME_TIMING_H
