#include "runtime/helpers.h"

extern "C"  __attribute__((always_inline)) uint64_t _mlir_ciface_timestamp_add_millis(uint64_t val,uint64_t millis) {
   return val+millis;
}