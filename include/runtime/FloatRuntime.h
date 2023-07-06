#ifndef RUNTIME_FLOATRUNTIME_H
#define RUNTIME_FLOATRUNTIME_H
#include "runtime/helpers.h"
namespace runtime {
struct FloatRuntime {
   static double sqrt(double);
   static double sin(double);
   static double cos(double);
   static double arcsin(double);
   static double exp(double);
   static double log(double);
   static double erf(double);
};

} // namespace runtime
#endif // RUNTIME_FLOATRUNTIME_H
