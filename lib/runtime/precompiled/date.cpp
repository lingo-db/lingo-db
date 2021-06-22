#include "runtime/helpers.h"

extern "C" uint64_t _mlir_ciface_timestamp_add_millis(uint64_t val,uint64_t millis) {
   return val+millis;
}