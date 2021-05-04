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

EXPORT void gdvXlargeMultiplyAndScaleDown(int64_t xHigh, uint64_t xLow, int64_t yHigh,
                                        uint64_t yLow, int32_t reduceScaleBy,
                                        int64_t* outHigh, uint64_t* outLow,
                                        bool* overflow);

EXPORT void gdvXlargeScaleUpAndDivide(int64_t xHigh, uint64_t xLow, int64_t yHigh,
                                    uint64_t yLow, int32_t increaseScaleBy,
                                    int64_t* outHigh, uint64_t* outLow, bool* overflow);

EXPORT void gdvXlargeMod(int64_t xHigh, uint64_t xLow, int32_t xScale, int64_t yHigh,
                    uint64_t yLow, int32_t yScale, int64_t* outHigh,
                    uint64_t* outLow);

EXPORT int32_t gdvXlargeCompare(int64_t xHigh, uint64_t xLow, int32_t xScale,
                           int64_t yHigh, uint64_t yLow, int32_t yScale);

EXPORT int32_t gdvFnDecFromString(int64_t context, const char* in, int32_t inLength,
                                      int32_t* precisionFromStr, int32_t* scaleFromStr,
                                      int64_t* decHighFromStr, uint64_t* decLowFromStr);

EXPORT char* gdvFnDecToString(int64_t context, int64_t xHigh, uint64_t xLow,
                           int32_t xScale, int32_t* decStrLen);
EXPORT int gdvFnTimeWithZone(int* timeFields, const char* zone, int zoneLen,
                                 int64_t* retTime);
EXPORT void gdvFnContextSetErrorMsg(int64_t contextPtr, const char* errMsg);
EXPORT uint8_t* gdvFnContextArenaMalloc(int64_t contextPtr, int32_t dataLen);

#endif // RUNTIME_RUNTIME_H
