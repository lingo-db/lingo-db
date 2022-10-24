#include "runtime/IntegerRuntime.h"
#include <cassert>
int64_t runtime::IntegerRuntime::round64(int64_t value, int64_t roundByScale) {
   assert(roundByScale>=0);
   return value;
}
int32_t runtime::IntegerRuntime::round32(int32_t value, int64_t roundByScale) {
   assert(roundByScale>=0);
   return value;
}
int16_t runtime::IntegerRuntime::round16(int16_t value, int64_t roundByScale) {
   assert(roundByScale>=0);
   return value;
}
int8_t runtime::IntegerRuntime::round8(int8_t value, int64_t roundByScale) {
   assert(roundByScale>=0);
   return value;
}
