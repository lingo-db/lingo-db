#include "lingodb/runtime/FloatRuntime.h"
#include <cassert>
#include <cmath>

double lingodb::runtime::FloatRuntime::sqrt(double x) {
   return ::sqrt(x);
}
double lingodb::runtime::FloatRuntime::sin(double x) {
   return ::sin(x);
}
double lingodb::runtime::FloatRuntime::cos(double x) {
   return ::cos(x);
}
double lingodb::runtime::FloatRuntime::arcsin(double x) {
   return ::asin(x);
}
double lingodb::runtime::FloatRuntime::log(double x) {
   return ::log(x);
}
double lingodb::runtime::FloatRuntime::exp(double x) {
   return ::exp(x);
}
double lingodb::runtime::FloatRuntime::erf(double x) {
   return ::erf(x);
}
double lingodb::runtime::FloatRuntime::pow(double x,double y) {
   return ::pow(x,y);
}