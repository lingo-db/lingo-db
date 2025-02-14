#ifndef LINGODB_RUNTIME_THREADLOCAL_H
#define LINGODB_RUNTIME_THREADLOCAL_H
#include "lingodb/scheduler/Scheduler.h"

#include <span>
namespace lingodb::runtime {
class ThreadLocal {
   uint8_t** values;
   uint8_t* (*initFn)(uint8_t*);
   uint8_t* arg;
   ThreadLocal(uint8_t* (*initFn)(uint8_t*), uint8_t* arg) : initFn(initFn), arg(arg) {
      values = new uint8_t*[lingodb::scheduler::getNumWorkers()];
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         values[i] = nullptr;
      }
   }

   public:
   uint8_t* getLocal();
   static ThreadLocal* create(uint8_t* (*initFn)(uint8_t*), uint8_t*);
   template <class T>
   std::span<T*> getThreadLocalValues() {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         if (values[i]) break;
         if (i == lingodb::scheduler::getNumWorkers() - 1) {
            values[i] = initFn(arg);
         }
      }
      return std::span<T*>(reinterpret_cast<T**>(values), lingodb::scheduler::getNumWorkers());
   }
   uint8_t* merge(void (*mergeFn)(uint8_t*, uint8_t*));
};
} // end namespace lingodb::runtime

#endif //LINGODB_RUNTIME_THREADLOCAL_H
