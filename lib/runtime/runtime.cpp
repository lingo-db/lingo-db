#include "runtime/runtime.h"
#include "arrow/util/decimal.h"
#include <iomanip>
#include <iostream>

EXPORT void dumpInt(bool null, int64_t val) {
   if (null) {
      std::cout << "int(NULL)" << std::endl;
   } else {
      std::cout << "int(" << val << ")" << std::endl;
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
EXPORT void dumpDate(bool null, uint32_t date) {
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
EXPORT void dumpInterval(bool null, uint64_t interval) {
   if (null) {
      std::cout << "interval(NULL)" << std::endl;
   } else {
      std::cout << "interval(" << interval << ")" << std::endl;
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