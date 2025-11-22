#include "lingodb/runtime/TryExcept.h"

void lingodb::runtime::TryExcept::run(void (*tryBlock)(void*, void*), void (*exceptBlock)(void*, void*), void* tryArg, void* exceptArg, void* res) {
   try {
      tryBlock(tryArg, res);
   } catch (...) {
      exceptBlock(exceptArg, res);
   }
}