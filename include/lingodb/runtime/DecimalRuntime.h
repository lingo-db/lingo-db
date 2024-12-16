#ifndef LINGODB_RUNTIME_DECIMALRUNTIME_H
#define LINGODB_RUNTIME_DECIMALRUNTIME_H
#include "lingodb/runtime/helpers.h"
namespace lingodb::runtime {
struct DecimalRuntime {
   static __int128 round(__int128 value, int64_t digits, int64_t scale);
};
} // namespace lingodb::runtime
#endif // LINGODB_RUNTIME_DECIMALRUNTIME_H
