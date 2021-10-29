#include "runtime/helpers.h"

#include <iostream>

#include <arrow/util/decimal.h>
#include <arrow/vendored/datetime.h>

EXPORT void rt_dump_int(bool null, int64_t val) {
   if (null) {
      std::cout << "int(NULL)" << std::endl;
   } else {
      std::cout << "int(" << val << ")" << std::endl;
   }
}
EXPORT void rt_dump_index(uint64_t val) {
      std::cout << "index(" << val << ")" << std::endl;
}
EXPORT void rt_dump_uint(bool null, uint64_t val) {
   if (null) {
      std::cout << "uint(NULL)" << std::endl;
   } else {
      std::cout << "uint(" << val << ")" << std::endl;
   }
}
EXPORT void rt_dump_bool(bool null, bool val) {
   if (null) {
      std::cout << "bool(NULL)" << std::endl;
   } else {
      std::cout << "bool(" << std::boolalpha << val << ")" << std::endl;
   }
}
EXPORT void rt_dump_decimal(bool null, uint64_t low, uint64_t high, int32_t scale) {
   if (null) {
      std::cout << "decimal(NULL)" << std::endl;
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
      std::cout << "decimal(" << decimalrep.ToString(scale) << ")" << std::endl;
   }
}
arrow_vendored::date::sys_days epoch = arrow_vendored::date::sys_days{arrow_vendored::date::jan / 1 / 1970};
EXPORT void rt_dump_date_day(bool null, uint32_t date) {
   if (null) {
      std::cout << "date(NULL)" << std::endl;
   } else {
      std::cout << "date(" << arrow_vendored::date::format("%F", epoch + arrow_vendored::date::days{date}) << ")" << std::endl;
   }
}
EXPORT void rt_dump_date_millisecond(bool null, int64_t date) {
   if (null) {
      std::cout << "date(NULL)" << std::endl;
   } else {
      std::cout << "date(" << arrow_vendored::date::format("%F", epoch + std::chrono::milliseconds{date}) << ")" << std::endl;
   }
}
template <class Unit>
void dump_timestamp(bool null, uint64_t date) {
   if (null) {
      std::cout << "timestamp(NULL)" << std::endl;
   } else {
      std::cout << "timestamp(" << arrow_vendored::date::format("%F %T", epoch + Unit{date}) << ")" << std::endl;
   }
}
EXPORT void rt_dump_timestamp_second(bool null, uint64_t date) {
   dump_timestamp<std::chrono::seconds>(null, date);
}
EXPORT void rt_dump_timestamp_millisecond(bool null, uint64_t date) {
   dump_timestamp<std::chrono::milliseconds>(null, date);
}
EXPORT void rt_dump_timestamp_microsecond(bool null, uint64_t date) {
   dump_timestamp<std::chrono::microseconds>(null, date);
}
EXPORT void rt_dump_timestamp_nanosecond(bool null, uint64_t date) {
   dump_timestamp<std::chrono::nanoseconds>(null, date);
}

EXPORT void rt_dump_interval_months(bool null, uint32_t interval) {
   if (null) {
      std::cout << "interval(NULL)" << std::endl;
   } else {
      std::cout << "interval(" << interval << " months)" << std::endl;
   }
}
EXPORT void rt_dump_interval_daytime(bool null, uint64_t interval) {
   if (null) {
      std::cout << "interval(NULL)" << std::endl;
   } else {
      std::cout << "interval(" << interval << " daytime)" << std::endl;
   }
}
EXPORT void rt_dump_float(bool null, double val) {
   if (null) {
      std::cout << "float(NULL)" << std::endl;
   } else {
      std::cout << "float(" << val << ")" << std::endl;
   }
}

EXPORT void rt_dump_string(bool null, runtime::Str string) {
   if (null) {
      std::cout << "string(NULL)" << std::endl;
   } else {
      std::cout << "string(\"" << string.str() << "\")" << std::endl;
   }
}
