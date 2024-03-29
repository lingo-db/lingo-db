#ifndef RUNTIME_DATERUNTIME_H
#define RUNTIME_DATERUNTIME_H
#include "runtime/helpers.h"
namespace runtime {
struct DateRuntime {
   static int64_t extractDay(int64_t date);
   static int64_t extractMonth(int64_t date);
   static int64_t extractYear(int64_t date);
   static int64_t subtractMonths(int64_t date, int64_t months);
   static int64_t addMonths(int64_t date, int64_t months);
   static int64_t dateDiffSeconds(int64_t start, int64_t end);
   static int64_t extractHour(int64_t date);
};
} // namespace runtime
#endif // RUNTIME_DATERUNTIME_H
