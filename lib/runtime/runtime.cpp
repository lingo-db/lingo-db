#include "runtime/runtime.h"
#include "arrow/util/decimal.h"
#include "arrow/vendored/datetime.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <iomanip>
#include <iostream>

EXPORT void dumpInt(bool null, int64_t val) {
   if (null) {
      std::cout << "int(NULL)" << std::endl;
   } else {
      std::cout << "int(" << val << ")" << std::endl;
   }
}
EXPORT void dumpUInt(bool null, uint64_t val) {
   if (null) {
      std::cout << "uint(NULL)" << std::endl;
   } else {
      std::cout << "uint(" << val << ")" << std::endl;
   }
}
EXPORT void dumpBool(bool null, bool val) {
   if (null) {
      std::cout << "bool(NULL)" << std::endl;
   } else {
      std::cout << "bool(" << std::boolalpha << val << ")" << std::endl;
   }
}
EXPORT void dumpDecimal(bool null, uint64_t low, uint64_t high, int32_t scale) {
   if (null) {
      std::cout << "decimal(NULL)" << std::endl;
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
      std::cout << "decimal(" << decimalrep.ToString(scale) << ")" << std::endl;
   }
}
EXPORT void dumpDate32(bool null, uint32_t date) {
   if (null) {
      std::cout << "date(NULL)" << std::endl;
   } else {
      time_t time = date;
      tm tmStruct;
      time *= 24 * 60 * 60;
      auto* x = gmtime_r(&time, &tmStruct);

      std::cout << "date(" << (x->tm_year + 1900) << "-" << std::setw(2) << std::setfill('0') << (x->tm_mon + 1) << "-" << std::setw(2) << std::setfill('0') << x->tm_mday << ")" << std::endl;
   }
}
EXPORT void dumpDate64(bool null, uint64_t date) {
   if (null) {
      std::cout << "date(NULL)" << std::endl;
   } else {
      time_t time = date;
      tm tmStruct;
      time /= 1000;
      auto* x = gmtime_r(&time, &tmStruct);

      std::cout << "date(" << (x->tm_year + 1900) << "-" << std::setw(2) << std::setfill('0') << (x->tm_mon + 1) << "-" << std::setw(2) << std::setfill('0') << x->tm_mday << ")" << std::endl;
   }
}
EXPORT void dumpTimestamp(bool null, uint64_t date) {
   if (null) {
      std::cout << "timestamp(NULL)" << std::endl;
   } else {
      time_t time = date;
      tm tmStruct;
      auto* x = gmtime_r(&time, &tmStruct);
      std::cout << "timestamp(" << (x->tm_year + 1900) << "-" << std::setw(2) << std::setfill('0') << (x->tm_mon + 1) << "-" << std::setw(2) << std::setfill('0') << x->tm_mday << " " << std::setw(2) << std::setfill('0') << x->tm_hour << ":" << std::setw(2) << std::setfill('0') << x->tm_min << ":" << std::setw(2) << std::setfill('0') << x->tm_sec << ")" << std::endl;
   }
}

EXPORT void dumpIntervalMonths(bool null, uint32_t interval) {
   if (null) {
      std::cout << "interval(NULL)" << std::endl;
   } else {
      std::cout << "interval(" << interval << " months)" << std::endl;
   }
}
EXPORT void dumpIntervalDaytime(bool null, uint64_t interval) {
   if (null) {
      std::cout << "interval(NULL)" << std::endl;
   } else {
      std::cout << "interval(" << interval << " daytime)" << std::endl;
   }
}
EXPORT void dumpFloat(bool null, double val) {
   if (null) {
      std::cout << "float(NULL)" << std::endl;
   } else {
      std::cout << "float(" << val << ")" << std::endl;
   }
}
EXPORT void dumpString(bool null, char* ptr, size_t len) {
   if (null) {
      std::cout << "string(NULL)" << std::endl;
   } else {
      std::cout << "string(\"" << std::string(ptr, len) << "\")" << std::endl;
   }
}
EXPORT uint32_t dateAdd(uint32_t date, uint32_t amount, TimeUnit unit) {
   //todo!!!
   time_t time = date;
   tm tmStruct;
   time *= 24 * 60 * 60;
   auto* x = gmtime_r(&time, &tmStruct);
   switch (unit) {
      case YEAR:
         x->tm_year += amount;
         return timegm(&tmStruct) / (24 * 60 * 60);
      case MONTH:
         x->tm_mon += amount;
         return timegm(&tmStruct) / (24 * 60 * 60);
      case DAY:
         time += 24 * 60 * 60 * amount;
         return time / (24 * 60 * 60);
      default:
         return date;
   }
}
EXPORT uint32_t dateSub(uint32_t date, uint32_t amount, TimeUnit unit) {
   //todo!!!
   time_t time = date;
   tm tmStruct;
   time *= 24 * 60 * 60;
   auto* x = gmtime_r(&time, &tmStruct);
   switch (unit) {
      case YEAR:
         x->tm_year -= amount;
         return timegm(&tmStruct) / (24 * 60 * 60);
      case MONTH:
         x->tm_mon -= amount;
         return timegm(&tmStruct) / (24 * 60 * 60);
      case DAY:
         time -= 24 * 60 * 60 * amount;
         return time / (24 * 60 * 60);
      default:
         return date;
   }
}
EXPORT uint32_t dateExtract(uint32_t date, TimeUnit unit) {
   //todo!!
   time_t time = date;
   tm tmStruct;
   time *= 24 * 60 * 60;
   auto* x = gmtime_r(&time, &tmStruct);
   switch (unit) {
      case YEAR:
         return x->tm_year + 1900;
      case MONTH:
         return x->tm_mon + 1;
      case DAY:
         return x->tm_mday;
      default:
         return 0;
   }
}

//gandiva

using arrow::BasicDecimal128;
using boost::multiprecision::int256_t;

namespace gandiva {
namespace internal {

// Convert to 256-bit integer from 128-bit decimal.
static int256_t convertToInt256(BasicDecimal128 in) {
   int256_t v = in.high_bits();
   v <<= 64;
   v |= in.low_bits();
   return v;
}

// Convert to 128-bit decimal from 256-bit integer.
// If there is an overflow, the output is undefined.
static BasicDecimal128 convertToDecimal128(int256_t in, bool* overflow) {
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

static int256_t getScaleMultiplier(int scale) {
   //DCHECK_GE(scale, 0);
   //DCHECK_LE(scale, kMaxLargeScale);

   return kLargeScaleMultipliers[scale];
}

// divide input by 10^reduce_by, and round up the fractional part.
static int256_t reduceScaleBy(int256_t in, int32_t reduceBy) {
   if (reduceBy == 0) {
      // nothing to do.
      return in;
   }

   int256_t divisor = getScaleMultiplier(reduceBy);
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
static int256_t increaseScaleBy(int256_t in, int32_t increaseBy) {
   //DCHECK_GE(increase_by, 0);
   //DCHECK_LE(increase_by, 2 * DecimalTypeUtil::kMaxPrecision);

   return in * getScaleMultiplier(increaseBy);
}

} // namespace internal
} // namespace gandiva

extern "C" {

void gdvXlargeMultiplyAndScaleDown(int64_t xHigh, uint64_t xLow, int64_t yHigh,
                                   uint64_t yLow, int32_t reduceScaleBy,
                                   int64_t* outHigh, uint64_t* outLow,
                                   bool* overflow) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};
   auto intermediateResult =
      gandiva::internal::convertToInt256(x) * gandiva::internal::convertToInt256(y);
   intermediateResult =
      gandiva::internal::reduceScaleBy(intermediateResult, reduceScaleBy);
   auto result = gandiva::internal::convertToDecimal128(intermediateResult, overflow);
   *outHigh = result.high_bits();
   *outLow = result.low_bits();
}

void gdvXlargeScaleUpAndDivide(int64_t xHigh, uint64_t xLow, int64_t yHigh,
                               uint64_t yLow, int32_t increaseScaleBy,
                               int64_t* outHigh, uint64_t* outLow,
                               bool* overflow) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};

   int256_t xLarge = gandiva::internal::convertToInt256(x);
   int256_t xLargeScaledUp =
      gandiva::internal::increaseScaleBy(xLarge, increaseScaleBy);
   int256_t yLarge = gandiva::internal::convertToInt256(y);
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
   auto result = gandiva::internal::convertToDecimal128(resultLarge, overflow);
   *outHigh = result.high_bits();
   *outLow = result.low_bits();
}

void gdvXlargeMod(int64_t xHigh, uint64_t xLow, int32_t xScale, int64_t yHigh,
                  uint64_t yLow, int32_t yScale, int64_t* outHigh,
                  uint64_t* outLow) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};

   int256_t xLarge = gandiva::internal::convertToInt256(x);
   int256_t yLarge = gandiva::internal::convertToInt256(y);
   if (xScale < yScale) {
      xLarge = gandiva::internal::increaseScaleBy(xLarge, yScale - xScale);
   } else {
      yLarge = gandiva::internal::increaseScaleBy(yLarge, xScale - yScale);
   }
   auto intermediateResult = xLarge % yLarge;
   bool overflow = false;
   auto result = gandiva::internal::convertToDecimal128(intermediateResult, &overflow);
   //DCHECK_EQ(overflow, false);

   *outHigh = result.high_bits();
   *outLow = result.low_bits();
}

int32_t gdvXlargeCompare(int64_t xHigh, uint64_t xLow, int32_t xScale,
                         int64_t yHigh, uint64_t yLow, int32_t yScale) {
   BasicDecimal128 x{xHigh, xLow};
   BasicDecimal128 y{yHigh, yLow};

   int256_t xLarge = gandiva::internal::convertToInt256(x);
   int256_t yLarge = gandiva::internal::convertToInt256(y);
   if (xScale < yScale) {
      xLarge = gandiva::internal::increaseScaleBy(xLarge, yScale - xScale);
   } else {
      yLarge = gandiva::internal::increaseScaleBy(yLarge, xScale - yScale);
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

EXPORT void gdvFnContextSetErrorMsg(int64_t contextPtr, const char* errMsg) {
   std::cout << "ERROR:" << std::string(errMsg) << "\n";
}
EXPORT uint8_t* gdvFnContextArenaMalloc(int64_t contextPtr, int32_t dataLen) {
   return static_cast<uint8_t*>(malloc(dataLen)); //todo
}

EXPORT int32_t gdvFnDecFromString(int64_t context, const char* in, int32_t inLength,
                                  int32_t* precisionFromStr, int32_t* scaleFromStr,
                                  int64_t* decHighFromStr, uint64_t* decLowFromStr) {
   arrow::Decimal128 dec;
   auto status = arrow::Decimal128::FromString(std::string(in, inLength), &dec,
                                               precisionFromStr, scaleFromStr);
   if (!status.ok()) {
      gdvFnContextSetErrorMsg(context, status.message().data());
      return -1;
   }
   *decHighFromStr = dec.high_bits();
   *decLowFromStr = dec.low_bits();
   return 0;
}

EXPORT char* gdvFnDecToString(int64_t context, int64_t xHigh, uint64_t xLow,
                              int32_t xScale, int32_t* decStrLen) {
   arrow::Decimal128 dec(arrow::BasicDecimal128(xHigh, xLow));
   std::string decStr = dec.ToString(xScale);
   *decStrLen = static_cast<int32_t>(decStr.length());
   char* ret = reinterpret_cast<char*>(gdvFnContextArenaMalloc(context, *decStrLen));
   if (ret == nullptr) {
      std::string errMsg = "Could not allocate memory for string: " + decStr;
      gdvFnContextSetErrorMsg(context, errMsg.data());
      return nullptr;
   }
   memcpy(ret, decStr.data(), *decStrLen);
   return ret;
}

// TODO : Do input validation or make sure the callers do that ?
EXPORT int gdvFnTimeWithZone(int* timeFields, const char* zone, int zoneLen,
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
