#include "runtime/helpers.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <arrow/util/decimal.h>
#include <arrow/vendored/datetime.h>
//gandiva

using arrow::BasicDecimal128;
using boost::multiprecision::int256_t;

namespace gandiva {
namespace internal {

// Convert to 256-bit integer from 128-bit decimal.
static int256_t convert_to_int_256(BasicDecimal128 in) {
   int256_t v = in.high_bits();
   v <<= 64;
   v |= in.low_bits();
   return v;
}

// Convert to 128-bit decimal from 256-bit integer.
// If there is an overflow, the output is undefined.
static BasicDecimal128 convert_to_decimal_128(int256_t in, bool* overflow) {
   BasicDecimal128 result;
   constexpr int256_t uinT64Mask = std::numeric_limits<uint64_t>::max();

   int256_t inAbs = abs(in);
   bool isNegative = in < 0;

   uint64_t low = (inAbs & uinT64Mask).convert_to<uint64_t>();
   inAbs >>= 64;
   uint64_t high = (inAbs & uinT64Mask).convert_to<uint64_t>();
   inAbs >>= 64;

   if (inAbs > 0) {
      // we've shifted in by 128-bit, so nothing should be left.
      *overflow = true;
   } else if (high > INT64_MAX) {
      // the high-bit must not be set (signed 128-bit).
      *overflow = true;
   } else {
      result = BasicDecimal128(static_cast<int64_t>(high), low);
      if (result > BasicDecimal128::GetMaxValue()) {
         *overflow = true;
      }
   }
   return isNegative ? -result : result;
}

static constexpr int32_t kMaxLargeScale = 2 * 38;

// Compute the scale multipliers once.
static std::array<int256_t, kMaxLargeScale + 1> kLargeScaleMultipliers =
   ([]() -> std::array<int256_t, kMaxLargeScale + 1> {
      std::array<int256_t, kMaxLargeScale + 1> values;
      values[0] = 1;
      for (int32_t idx = 1; idx <= kMaxLargeScale; idx++) {
         values[idx] = values[idx - 1] * 10;
      }
      return values;
   })();

static int256_t get_scale_multiplier(int scale) {
   //DCHECK_GE(scale, 0);
   //DCHECK_LE(scale, kMaxLargeScale);

   return kLargeScaleMultipliers[scale];
}

// divide input by 10^reduce_by, and round up the fractional part.
static int256_t reduce_scale_by(int256_t in, int32_t reduceBy) {
   if (reduceBy == 0) {
      // nothing to do.
      return in;
   }

   int256_t divisor = get_scale_multiplier(reduceBy);
   //DCHECK_GT(divisor, 0);
   //DCHECK_EQ(divisor % 2, 0);  // multiple of 10.
   auto result = in / divisor;
   auto remainder = in % divisor;
   // round up (same as BasicDecimal128::ReduceScaleBy)
   if (abs(remainder) >= (divisor >> 1)) {
      result += (in > 0 ? 1 : -1);
   }
   return result;
}

// multiply input by 10^increase_by.
static int256_t increase_scale_by(int256_t in, int32_t increaseBy) {
   //DCHECK_GE(increase_by, 0);
   //DCHECK_LE(increase_by, 2 * DecimalTypeUtil::kMaxPrecision);

   return in * get_scale_multiplier(increaseBy);
}

} // namespace internal
} // namespace gandiva

extern "C" {

void gdv_xlarge_multiply_and_scale_down(int64_t xHigh, uint64_t xLow, int64_t yHigh,
                                        uint64_t yLow, int32_t reduce_scale_by,
                                        int64_t* outHigh, uint64_t* outLow,
                                        bool* overflow) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};
   auto intermediateResult =
      gandiva::internal::convert_to_int_256(x) * gandiva::internal::convert_to_int_256(y);
   intermediateResult =
      gandiva::internal::reduce_scale_by(intermediateResult, reduce_scale_by);
   auto result = gandiva::internal::convert_to_decimal_128(intermediateResult, overflow);
   *outHigh = result.high_bits();
   *outLow = result.low_bits();
}

void gdv_xlarge_scale_up_and_divide(int64_t xHigh, uint64_t xLow, int64_t yHigh,
                                    uint64_t yLow, int32_t increase_scale_by,
                                    int64_t* outHigh, uint64_t* outLow,
                                    bool* overflow) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};

   int256_t xLarge = gandiva::internal::convert_to_int_256(x);
   int256_t xLargeScaledUp =
      gandiva::internal::increase_scale_by(xLarge, increase_scale_by);
   int256_t yLarge = gandiva::internal::convert_to_int_256(y);
   int256_t resultLarge = xLargeScaledUp / yLarge;
   int256_t remainderLarge = xLargeScaledUp % yLarge;

   // Since we are scaling up and then, scaling down, round-up the result (+1 for +ve,
   // -1 for -ve), if the remainder is >= 2 * divisor.
   if (abs(2 * remainderLarge) >= abs(yLarge)) {
      // x +ve and y +ve, result is +ve =>   (1 ^ 1)  + 1 =  0 + 1 = +1
      // x +ve and y -ve, result is -ve =>  (-1 ^ 1)  + 1 = -2 + 1 = -1
      // x +ve and y -ve, result is -ve =>   (1 ^ -1) + 1 = -2 + 1 = -1
      // x -ve and y -ve, result is +ve =>  (-1 ^ -1) + 1 =  0 + 1 = +1
      resultLarge += (x.Sign() ^ y.Sign()) + 1;
   }
   auto result = gandiva::internal::convert_to_decimal_128(resultLarge, overflow);
   *outHigh = result.high_bits();
   *outLow = result.low_bits();
}

void gdv_xlarge_mod(int64_t xHigh, uint64_t xLow, int32_t xScale, int64_t yHigh,
                    uint64_t yLow, int32_t yScale, int64_t* outHigh,
                    uint64_t* outLow) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};

   int256_t xLarge = gandiva::internal::convert_to_int_256(x);
   int256_t yLarge = gandiva::internal::convert_to_int_256(y);
   if (xScale < yScale) {
      xLarge = gandiva::internal::increase_scale_by(xLarge, yScale - xScale);
   } else {
      yLarge = gandiva::internal::increase_scale_by(yLarge, xScale - yScale);
   }
   auto intermediateResult = xLarge % yLarge;
   bool overflow = false;
   auto result = gandiva::internal::convert_to_decimal_128(intermediateResult, &overflow);
   //DCHECK_EQ(overflow, false);

   *outHigh = result.high_bits();
   *outLow = result.low_bits();
}

int32_t gdv_xlarge_compare(int64_t xHigh, uint64_t xLow, int32_t xScale,
                           int64_t yHigh, uint64_t yLow, int32_t yScale) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};

   int256_t xLarge = gandiva::internal::convert_to_int_256(x);
   int256_t yLarge = gandiva::internal::convert_to_int_256(y);
   if (xScale < yScale) {
      xLarge = gandiva::internal::increase_scale_by(xLarge, yScale - xScale);
   } else {
      yLarge = gandiva::internal::increase_scale_by(yLarge, xScale - yScale);
   }

   if (xLarge == yLarge) {
      return 0;
   } else if (xLarge < yLarge) {
      return -1;
   } else {
      return 1;
   }
}

} // extern "C"

EXPORT void gdv_fn_context_set_error_msg(int64_t contextPtr, const char* errMsg) {
   std::cout << "ERROR:" << std::string(errMsg) << "\n";
}
EXPORT uint8_t* gdv_fn_context_arena_malloc(int64_t contextPtr, int32_t dataLen) {
   return static_cast<uint8_t*>(malloc(dataLen)); //todo
}

EXPORT int32_t gdv_fn_dec_from_string(int64_t context, const char* in, int32_t inLength,
                                      int32_t* precisionFromStr, int32_t* scaleFromStr,
                                      int64_t* decHighFromStr, uint64_t* decLowFromStr) {
   arrow::Decimal128 dec;
   auto status = arrow::Decimal128::FromString(std::string(in, inLength), &dec,
                                               precisionFromStr, scaleFromStr);
   if (!status.ok()) {
      gdv_fn_context_set_error_msg(context, status.message().data());
      return -1;
   }
   *decHighFromStr = dec.high_bits();
   *decLowFromStr = dec.low_bits();
   return 0;
}

EXPORT char* gdv_fn_dec_to_string(int64_t context, int64_t xHigh, uint64_t xLow,
                                  int32_t xScale, int32_t* decStrLen) {
   arrow::Decimal128 dec(arrow::BasicDecimal128(xHigh, xLow));
   std::string decStr = dec.ToString(xScale);
   *decStrLen = static_cast<int32_t>(decStr.length());
   char* ret = reinterpret_cast<char*>(gdv_fn_context_arena_malloc(context, *decStrLen));
   if (ret == nullptr) {
      std::string errMsg = "Could not allocate memory for string: " + decStr;
      gdv_fn_context_set_error_msg(context, errMsg.data());
      return nullptr;
   }
   memcpy(ret, decStr.data(), *decStrLen);
   return ret;
}

// TODO : Do input validation or make sure the callers do that ?
EXPORT int gdv_fn_time_with_zone(int* timeFields, const char* zone, int zoneLen,
                                 int64_t* retTime) {
   using arrow_vendored::date::day;
   using arrow_vendored::date::local_days;
   using arrow_vendored::date::locate_zone;
   using arrow_vendored::date::month;
   using arrow_vendored::date::time_zone;
   using arrow_vendored::date::year;
   using std::chrono::hours;
   using std::chrono::milliseconds;
   using std::chrono::minutes;
   using std::chrono::seconds;

   enum TimeFields {
      kYear,
      kMonth,
      kDay,
      kHours,
      kMinutes,
      kSeconds,
      kSubSeconds,
      kDisplacementHours,
      kDisplacementMinutes,
      kMax
   };
   try {
      const time_zone* tz = locate_zone(std::string(zone, zoneLen));
      *retTime = tz->to_sys(local_days(year(timeFields[TimeFields::kYear]) /
                                       month(timeFields[TimeFields::kMonth]) /
                                       day(timeFields[TimeFields::kDay])) +
                            hours(timeFields[TimeFields::kHours]) +
                            minutes(timeFields[TimeFields::kMinutes]) +
                            seconds(timeFields[TimeFields::kSeconds]) +
                            milliseconds(timeFields[TimeFields::kSubSeconds]))
                    .time_since_epoch()
                    .count();
   } catch (...) {
      return EINVAL;
   }

   return 0;
}
