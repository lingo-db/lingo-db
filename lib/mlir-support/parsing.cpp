#include "mlir-support/parsing.h"
#include "arrow/util/decimal.h"
#include "arrow/util/value_parsing.h"
int32_t parseDate32(std::string str) {
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

std::pair<uint64_t, uint64_t> support::getDecimalScaleMultiplier(int32_t scale) {
   auto decimalrep = arrow::Decimal128::GetScaleMultiplier(scale);
   return {decimalrep.low_bits(), (uint64_t) decimalrep.high_bits()};
}
std::pair<uint64_t, uint64_t> support::parseDecimal(std::string str, int32_t reqScale) {
   int32_t precision;
   int32_t scale;
   arrow::Decimal128 decimalrep;
   if (!arrow::Decimal128::FromString(str, &decimalrep, &precision, &scale).ok()) {
      assert(false && "could not parse decimal const");
   }
   auto x = decimalrep.Rescale(scale, reqScale);
   decimalrep = x.ValueOrDie();
   uint64_t low = decimalrep.low_bits();
   uint64_t high = decimalrep.high_bits();
   return {low, high};
}

std::variant<int64_t, double, std::string> parseInt(std::variant<int64_t, double, std::string> val) {
   int64_t res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      res = std::get<double>(val);
   } else {
      res = std::stoll(std::get<std::string>(val));
   }
   return res;
}
std::variant<int64_t, double, std::string> parseInterval(std::variant<int64_t, double, std::string> val) {
   int64_t res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      throw std::runtime_error("can not parse interval from double");
   } else {
      res = std::stoll(std::get<std::string>(val));
      if (std::get<std::string>(val).ends_with("days")) {
         res *= 24 * 60 * 60 * 1000000000ll;
      }
   }
   return res;
}
std::variant<int64_t, double, std::string> parseDouble(std::variant<int64_t, double, std::string> val) {
   double res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      res = std::get<double>(val);
   } else {
      res = std::stod(std::get<std::string>(val));
   }
   return res;
}
std::variant<int64_t, double, std::string> parseBool(std::variant<int64_t, double, std::string> val) {
   bool res;
   if (std::holds_alternative<int64_t>(val)) {
      res = std::get<int64_t>(val);
   } else if (std::holds_alternative<double>(val)) {
      throw std::runtime_error("can not parse bool from double");
   } else {
      auto str = std::get<std::string>(val);
      if (str == "true" || str == "t") {
         res = true;
      } else if (str == "false" || str == "f") {
         res = false;
      } else {
         throw std::runtime_error("can not parse bool from value: " + str);
      }
   }
   return static_cast<int64_t>(res);
}
std::variant<int64_t, double, std::string> parseString(std::variant<int64_t, double, std::string> val, bool acceptInts = false) {
   std::string str;
   if (std::holds_alternative<int64_t>(val)) {
      if (acceptInts) {
         str = std::to_string(std::get<int64_t>(val));
      } else {
         throw std::runtime_error("can not parse string from int: " + str);
      }
   } else if (std::holds_alternative<double>(val)) {
      throw std::runtime_error("can not parse string from double: " + str);
   } else {
      str = std::get<std::string>(val);
   }
   return str;
}
std::variant<int64_t, double, std::string> parseDate(std::variant<int64_t, double, std::string> val, bool parse64 = false) {
   if (!std::holds_alternative<std::string>(val)) {
      throw std::runtime_error("can not parse date");
   }
   std::string str = std::get<std::string>(val);
   int64_t parsed = parseDate32(str);
   int64_t date64 = parsed * 24 * 60 * 60 * 1000000000ll;
   return date64;
}
std::variant<int64_t, double, std::string> toI64(std::variant<int64_t, double, std::string> val) {
   if (std::holds_alternative<std::string>(val)) {
      int64_t res = 0;
      auto str = std::get<std::string>(val);
      memcpy(&res, str.data(), std::min(sizeof(res), str.size()));
      return res;
   }
   return val;
}
std::variant<int64_t, double, std::string> parseTimestamp(std::variant<int64_t, double, std::string> val, support::TimeUnit unit) {
   if (!std::holds_alternative<std::string>(val)) {
      throw std::runtime_error("can not parse timestamp");
   }
   std::string str = std::get<std::string>(val);
   int64_t res;
   arrow::internal::ParseValue<arrow::TimestampType>(arrow::TimestampType(convertTimeUnit(unit)), str.data(), str.length(), &res);
   return res;
}
std::variant<int64_t, double, std::string> support::parse(std::variant<int64_t, double, std::string> val, arrow::Type::type type, uint32_t param1, uint32_t param2) {
   switch (type) {
      case arrow::Type::type::INT8:
      case arrow::Type::type::INT16:
      case arrow::Type::type::INT32:
      case arrow::Type::type::INT64:
      case arrow::Type::type::UINT8:
      case arrow::Type::type::UINT16:
      case arrow::Type::type::UINT32:
      case arrow::Type::type::UINT64:
      case arrow::Type::type::INTERVAL_DAY_TIME:
      case arrow::Type::type::INTERVAL_MONTHS:
         return parseInterval(val);
      case arrow::Type::type::BOOL: return parseBool(val);
      case arrow::Type::type::HALF_FLOAT:
      case arrow::Type::type::FLOAT:
      case arrow::Type::type::DOUBLE: return parseDouble(val);
      case arrow::Type::type::FIXED_SIZE_BINARY: return toI64(parseString(val));
      case arrow::Type::type::DECIMAL128: return parseString(val, true);
      case arrow::Type::type::STRING: return parseString(val,true);
      case arrow::Type::type::DATE32: return parseDate(val, false);
      case arrow::Type::type::DATE64: return parseDate(val, true);
      case arrow::Type::type::TIMESTAMP: return parseTimestamp(val, static_cast<TimeUnit>(param1));
      default:
         throw std::runtime_error("could not parse");
   }
}
