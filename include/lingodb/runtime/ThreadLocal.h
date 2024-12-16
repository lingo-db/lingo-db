#ifndef LINGODB_RUNTIME_THREADLOCAL_H
#define LINGODB_RUNTIME_THREADLOCAL_H
#include <oneapi/tbb.h>
namespace lingodb::runtime {
class ThreadLocal {
   tbb::enumerable_thread_specific<uint8_t*> tls;

   ThreadLocal(uint8_t* (*initFn)(uint8_t*),uint8_t* arg) : tls([initFn,arg](){return initFn(arg);}) {
   }
   public:
   uint8_t* getLocal();
   static ThreadLocal* create(uint8_t* (*initFn)(uint8_t*),uint8_t*);
   const tbb::enumerable_thread_specific<uint8_t*>& getTls() {
      if(tls.empty()){
         tls.local();
      }
      return tls;
   }
   uint8_t* merge(void (*mergeFn)(uint8_t*, uint8_t*));
};
} // end namespace lingodb::runtime

#endif //LINGODB_RUNTIME_THREADLOCAL_H
