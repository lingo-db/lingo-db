#include "runtime/ThreadLocal.h"

uint8_t* runtime::ThreadLocal::getLocal() {
   auto local = tls.local();
   return local;
}
runtime::ThreadLocal* runtime::ThreadLocal::create(uint8_t* (*initFn)()) {
   return new ThreadLocal(initFn);
}
