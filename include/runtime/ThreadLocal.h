#ifndef RUNTIME_THREADLOCAL_H
#define RUNTIME_THREADLOCAL_H
#include <oneapi/tbb.h>
namespace runtime {
class ThreadLocal {
   tbb::enumerable_thread_specific<uint8_t*> tls;

   ThreadLocal(uint8_t* (*initFn)()) : tls(initFn) {
   }
   public:
   uint8_t* getLocal();
   static ThreadLocal* create(uint8_t* (*initFn)());
   const tbb::enumerable_thread_specific<uint8_t*>& getTls() {
      if(tls.empty()){
         tls.local();
      }
      return tls;
   }
};
} // end namespace runtime

#endif //RUNTIME_THREADLOCAL_H
