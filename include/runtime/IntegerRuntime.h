#ifndef RUNTIME_INTEGERRUNTIME_H
#define RUNTIME_INTEGERRUNTIME_H
#include "runtime/helpers.h"
namespace runtime {
struct IntegerRuntime {
   static int64_t round64(int64_t value, int64_t roundByScale);
   static int32_t round32(int32_t value, int64_t roundByScale);
   static int16_t round16(int16_t value, int64_t roundByScale);
   static int8_t round8(int8_t value, int64_t roundByScale);
   static int64_t sqrt(int64_t);
   static int64_t randomInRange(int64_t from,int64_t to);
};
} // namespace runtime
#endif // RUNTIME_INTEGERRUNTIME_H
