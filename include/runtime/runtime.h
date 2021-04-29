#ifndef DB_DIALECTS_RUNTIME_H
#define DB_DIALECTS_RUNTIME_H

#include <cstddef>
#include <cstdint>
#define EXPORT extern "C" __attribute__((visibility("default")))

EXPORT void dumpInt(bool null, int64_t val);
EXPORT void dumpBool(bool null, bool val);
EXPORT void dumpDecimal(bool null, uint64_t low, uint64_t high, int32_t scale);
EXPORT void dumpDate(bool null, uint32_t date);
EXPORT void dumpTimestamp(bool null, uint64_t date);
EXPORT void dumpInterval(bool null, uint64_t interval);
EXPORT void dumpFloat(bool null, double val);
EXPORT void dumpString(bool null, char* ptr, size_t len);

#endif //DB_DIALECTS_RUNTIME_H
