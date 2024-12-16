#include "lingodb/runtime/IntegerRuntime.h"
#include <cassert>
#include <cmath>
#include <random>
int64_t lingodb::runtime::IntegerRuntime::round64(int64_t value, int64_t roundByScale) {
   assert(roundByScale >= 0);
   return value;
}
int32_t lingodb::runtime::IntegerRuntime::round32(int32_t value, int64_t roundByScale) {
   assert(roundByScale >= 0);
   return value;
}
int16_t lingodb::runtime::IntegerRuntime::round16(int16_t value, int64_t roundByScale) {
   assert(roundByScale >= 0);
   return value;
}
int8_t lingodb::runtime::IntegerRuntime::round8(int8_t value, int64_t roundByScale) {
   assert(roundByScale >= 0);
   return value;
}

int64_t lingodb::runtime::IntegerRuntime::sqrt(int64_t value) {
   return std::sqrt(value);
}

int64_t lingodb::runtime::IntegerRuntime::randomInRange(int64_t from, int64_t to) {
   static std::mt19937 gen(0);
   std::uniform_int_distribution<> distr(from, to - 1);
   return distr(gen);
}
