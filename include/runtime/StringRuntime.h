#ifndef RUNTIME_STRINGRUNTIME_H
#define RUNTIME_STRINGRUNTIME_H
#include "runtime/helpers.h"
namespace runtime {
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
   static float toFloat32(VarLen32 str);
   static double toFloat64(VarLen32 str);
   static __int128 toDecimal(VarLen32 str, unsigned reqScale);
   static VarLen32 fromInt(int64_t);
   static VarLen32 fromFloat32(float);
   static VarLen32 fromFloat64(double);
   static VarLen32 fromChar(uint64_t, size_t bytes);
   static VarLen32 fromDecimal(__int128, uint32_t scale);
   static VarLen32 substr(VarLen32 str, size_t from,size_t to);
};
} // namespace runtime
#endif // RUNTIME_STRINGRUNTIME_H
