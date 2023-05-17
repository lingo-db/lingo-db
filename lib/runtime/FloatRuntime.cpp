#include "runtime/FloatRuntime.h"
#include <cassert>
#include <cmath>

double runtime::FloatRuntime::sqrt(double x) {
   return ::sqrt(x);
}
double runtime::FloatRuntime::sin(double x) {
   return ::sin(x);
}
double runtime::FloatRuntime::cos(double x) {
   return ::cos(x);
}
double runtime::FloatRuntime::arcsin(double x) {
   return ::asin(x);
}
double runtime::FloatRuntime::log(double x) {
   return ::log(x);
}
double runtime::FloatRuntime::exp(double x) {
   return ::exp(x);
}
double runtime::FloatRuntime::erf(double x) {
   return ::erf(x);
}