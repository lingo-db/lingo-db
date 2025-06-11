#ifndef LINGODB_RUNTIME_STRINGRUNTIME_H
#define LINGODB_RUNTIME_STRINGRUNTIME_H
#include "lingodb/runtime/helpers.h"
namespace lingodb::runtime {
struct StringRuntime {
   static bool compareEq(VarLen32 l, VarLen32 r);
   static bool compareNEq(VarLen32 l, VarLen32 r);
   static bool compareLt(VarLen32 l, VarLen32 r);
   static bool compareGt(VarLen32 l, VarLen32 r);
   static bool compareLte(VarLen32 l, VarLen32 r);
   static bool compareGte(VarLen32 l, VarLen32 r);
   static bool like(VarLen32 l, VarLen32 r);
   static bool startsWith(VarLen32 str, VarLen32 substr);
   static bool endsWith(VarLen32 str, VarLen32 substr);
   static int64_t toInt(VarLen32 str);
   static int64_t len(VarLen32 str);
   static size_t nextChar(VarLen32 str, size_t position);
   static float toFloat32(VarLen32 str);
   static double toFloat64(VarLen32 str);
   static __int128 toDecimal(VarLen32 str, int32_t reqScale);
   static int64_t toDate(VarLen32 str);
   static int64_t toTimestamp(VarLen32 str);
   static int32_t toChar(VarLen32 str);
   static VarLen32 fromDate(int64_t);
   static VarLen32 fromTimestamp(int64_t);
   static VarLen32 fromBool(bool);
   static VarLen32 fromInt(int64_t);
   static VarLen32 fromFloat32(float);
   static VarLen32 fromFloat64(double);
   static VarLen32 fromChar(uint32_t);
   static VarLen32 fromDecimal(__int128, int32_t scale);
   static VarLen32 substr(VarLen32 str, int64_t from, int64_t len);
   static VarLen32 regexpReplace(VarLen32 str, VarLen32 pattern, VarLen32 replace);
   static VarLen32 toUpper(VarLen32 str);
   static VarLen32 concat(VarLen32 a, VarLen32 b);
   static size_t findMatch(VarLen32 str, VarLen32 needle, size_t start, size_t end);
   static size_t findNext(VarLen32 str, VarLen32 needle, size_t start);
};
} // namespace lingodb::runtime
#endif // LINGODB_RUNTIME_STRINGRUNTIME_H
