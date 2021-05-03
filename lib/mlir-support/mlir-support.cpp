#include "mlir-support/mlir-support.h"
#include "arrow/util/decimal.h"
#include "arrow/util/value_parsing.h"
int32_t support::parseDate32(std::string str) {
   int32_t res;
   arrow::internal::ParseValue<arrow::Date32Type>(str.data(), str.length(), &res);
   return res;
}
int64_t support::parseTimestamp(std::string str) {
   int64_t res;
   arrow::internal::ParseValue<arrow::TimestampType>(arrow::TimestampType(), str.data(), str.length(), &res);
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
      //todo
   }
   auto x = decimalrep.Rescale(scale, reqScale);
   decimalrep = x.ValueUnsafe();
   return {decimalrep.low_bits(), decimalrep.high_bits()};
}