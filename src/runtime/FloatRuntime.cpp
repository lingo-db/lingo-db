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
double lingodb::runtime::FloatRuntime::arccos(double x) {
   return ::acos(x);
}
double lingodb::runtime::FloatRuntime::arctan2(double y, double x) {
   return ::atan2(y, x);
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
double lingodb::runtime::FloatRuntime::pow(double x, double y) {
   return ::pow(x, y);
}
int64_t lingodb::runtime::FloatRuntime::ceil(double x) {
   return ::ceil(x);
}

namespace {
double round_half_even(double x) {
   if (!std::isfinite(x) || x == 0.0) return x; // keep NaN/Inf and signed zero

   double ipart;
   double frac = std::modf(x, &ipart); // frac has same sign as x

   // If not a tie (|frac| != 0.5), do ordinary nearest with halves handled below
   double afrac = std::fabs(frac);
   if (afrac < 0.5) return ipart;
   if (afrac > 0.5) return ipart + std::copysign(1.0, frac);

   // Tie case: exactly ±0.5 -> round to even
   double imod2 = std::fmod(std::fabs(ipart), 2.0);
   bool is_even = (imod2 == 0.0);
   if (is_even) return ipart;
   return ipart + std::copysign(1.0, frac);
}
} // namespace

double lingodb::runtime::FloatRuntime::round(double x, int64_t ndigits) {
   if (!std::isfinite(x) || x == 0.0) return x;

   if (ndigits == 0) {
      return round_half_even(x);
   }
   // For large |ndigits|, scaling by 10^ndigits can overflow/underflow.
   if (ndigits > 308) return x; // beyond double's decimal exponent range
   if (ndigits < -308) return std::copysign(0.0, x);

   double scale = std::pow(10.0, static_cast<double>(ndigits));
   if (!std::isfinite(scale) || scale == 0.0) {
      return (ndigits > 0) ? x : std::copysign(0.0, x);
   }
   double y = x * scale;
   if (!std::isfinite(y)) {
      return (ndigits > 0) ? x : std::copysign(0.0, x);
   }
   double r = round_half_even(y);
   return r / scale;
}