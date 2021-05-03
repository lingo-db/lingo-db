#ifndef RUNTIME_RUNTIME_H
#define RUNTIME_RUNTIME_H

#include <cstddef>
#include <cstdint>
#define EXPORT extern "C" __attribute__((visibility("default")))

EXPORT void dumpInt(bool null, int64_t val);
EXPORT void dumpUInt(bool null, uint64_t val);
EXPORT void dumpBool(bool null, bool val);
EXPORT void dumpDecimal(bool null, uint64_t low, uint64_t high, int32_t scale);
EXPORT void dumpDate(bool null, uint32_t date);
EXPORT void dumpTimestamp(bool null, uint64_t date);
EXPORT void dumpIntervalMonths(bool null, uint32_t interval);
EXPORT void dumpIntervalDaytime(bool null, uint64_t interval);

EXPORT void dumpFloat(bool null, double val);
EXPORT void dumpString(bool null, char* ptr, size_t len);

enum TimeUnit : uint8_t {
   YEAR,
   MONTH,
   DAY,
   HOUR,
   MINUTES,
   SECONDS,
   UNKNOWN
};
EXPORT uint32_t dateAdd(uint32_t,uint32_t amount,TimeUnit unit);
EXPORT uint32_t dateSub(uint32_t,uint32_t amount,TimeUnit unit);
EXPORT uint32_t dateExtract(uint32_t,TimeUnit unit);

#endif // RUNTIME_RUNTIME_H
