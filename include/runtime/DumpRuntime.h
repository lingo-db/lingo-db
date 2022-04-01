#ifndef RUNTIME_DUMPRUNTIME_H
#define RUNTIME_DUMPRUNTIME_H
#include "runtime/helpers.h"
#include <cstdint>
namespace runtime {
struct DumpRuntime {
   static void dumpIndex(uint64_t val);
   static void dumpInt(bool null, int64_t val);
   static void dumpUInt(bool null, uint64_t val);
   static void dumpBool(bool null, bool val);
   static void dumpDecimal(bool null, uint64_t low, uint64_t high, int32_t scale);
   static void dumpDate(bool null, int64_t date);
   static void dumpTimestampSecond(bool null, uint64_t date);
   static void dumpTimestampMilliSecond(bool null, uint64_t date);
   static void dumpTimestampMicroSecond(bool null, uint64_t date);
   static void dumpTimestampNanoSecond(bool null, uint64_t date);
   static void dumpIntervalMonths(bool null, uint32_t interval);
   static void dumpIntervalDaytime(bool null, uint64_t interval);
   static void dumpFloat(bool null, double val);
   static void dumpString(bool null, runtime::VarLen32 string);
   static void dumpChar(bool null, uint64_t val, size_t bytes);
};
} // namespace runtime
#endif // RUNTIME_DUMPRUNTIME_H
