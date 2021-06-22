#include "mlir-support/mlir-support.h"
#include "arrow/util/decimal.h"
#include "arrow/util/value_parsing.h"
int32_t support::parseDate32(std::string str) {
   int32_t res;
   arrow::internal::ParseValue<arrow::Date32Type>(str.data(), str.length(), &res);
   return res;
}
arrow::TimeUnit::type convertTimeUnit(support::TimeUnit unit) {
   switch (unit) {
      case support::TimeUnit::SECOND: return arrow::TimeUnit::SECOND;
      case support::TimeUnit::MILLI: return arrow::TimeUnit::MILLI;
      case support::TimeUnit::MICRO: return arrow::TimeUnit::MICRO;
      case support::TimeUnit::NANO: return arrow::TimeUnit::NANO;
   }
   return arrow::TimeUnit::SECOND;
}
int64_t support::parseTimestamp(std::string str, TimeUnit unit) {
   int64_t res;
   arrow::internal::ParseValue<arrow::TimestampType>(arrow::TimestampType(convertTimeUnit(unit)), str.data(), str.length(), &res);
   return res;
}
std::pair<uint64_t, uint64_t> support::getDecimalScaleMultiplier(unsigned scale) {
   auto decimalrep = arrow::Decimal128::GetScaleMultiplier(scale);
   return {decimalrep.low_bits(), (uint64_t) decimalrep.high_bits()};
}
std::pair<uint64_t, uint64_t> support::parseDecimal(std::string str, unsigned reqScale) {
   int32_t precision;
   int32_t scale;
   arrow::Decimal128 decimalrep;
   if (arrow::Decimal128::FromString(str, &decimalrep, &precision, &scale) != arrow::Status::OK()) {
      assert(false&&"could not parse decimal const");
   }
   auto x = decimalrep.Rescale(scale, reqScale);
   decimalrep = x.ValueOrDie();
   uint64_t low = decimalrep.low_bits();
   uint64_t high = decimalrep.high_bits();
   return {low, high};
}