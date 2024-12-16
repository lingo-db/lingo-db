#ifndef LINGODB_RUNTIME_FLOATRUNTIME_H
#define LINGODB_RUNTIME_FLOATRUNTIME_H
#include "lingodb/runtime/helpers.h"
namespace lingodb::runtime {
struct FloatRuntime {
   static double sqrt(double);
   static double sin(double);
   static double cos(double);
   static double arcsin(double);
   static double exp(double);
   static double log(double);
   static double erf(double);
   static double pow(double, double);
};

} // namespace lingodb::runtime
#endif // LINGODB_RUNTIME_FLOATRUNTIME_H
