#ifndef RUNTIME_DECIMALRUNTIME_H
#define RUNTIME_DECIMALRUNTIME_H
#include "runtime/helpers.h"
namespace runtime {
struct DecimalRuntime {
   static __int128 round(__int128 value, int64_t digits, int64_t scale);
};
} // namespace runtime
#endif // RUNTIME_DECIMALRUNTIME_H
