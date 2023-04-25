#include "runtime/ThreadLocal.h"

uint8_t* runtime::ThreadLocal::getLocal() {
   return tls.local();
}
runtime::ThreadLocal* runtime::ThreadLocal::create(uint8_t* (*initFn)()) {
   return new ThreadLocal(initFn);
}
