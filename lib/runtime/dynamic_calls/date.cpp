#include "arrow/vendored/datetime/date.h"
#include "runtime/helpers.h"
#include <iostream>
//adapted from apache gandiva
//source: https://github.com/apache/arrow/blob/3da66003ab2543c231fdf6551c2eb886f9a7e68f/cpp/src/gandiva/precompiled/epoch_time_point.h
//Apache-2.0 License
namespace date = arrow_vendored::date;
class DateHelper {
   public:
   explicit DateHelper(std::chrono::nanoseconds nanos_since_epoch)
      : tp_(nanos_since_epoch) {}

   explicit DateHelper(int64_t nanoseconds_since_epoch)
      : DateHelper(std::chrono::nanoseconds(nanoseconds_since_epoch)) {}

   int64_t TmYear() const { return static_cast<int>(YearMonthDay().year()) - 1900; }

   int64_t TmMon() const { return static_cast<unsigned int>(YearMonthDay().month()) - 1; }

   int64_t TmYday() const {
      auto to_days = date::floor<date::days>(tp_);
      auto first_day_in_year = date::sys_days{
         YearMonthDay().year() / date::jan / 1};
      return (to_days - first_day_in_year).count();
   }

   int64_t TmMday() const { return static_cast<unsigned int>(YearMonthDay().day()); }

   int64_t TmWday() const {
      auto to_days = date::floor<date::days>(tp_);
      return (date::weekday{to_days} - // NOLINT
              date::Sunday)
         .count();
   }

   int64_t TmHour() const { return static_cast<int>(TimeOfDay().hours().count()); }

   int64_t TmMin() const { return static_cast<int>(TimeOfDay().minutes().count()); }

   int64_t TmSec() const {
      return static_cast<int>(TimeOfDay().seconds().count());
   }

   DateHelper AddMonths(int num_months) const {
      auto ymd = YearMonthDay() + date::months(num_months);
      return DateHelper((date::sys_days{ymd} + // NOLINT
                         TimeOfDay().to_duration())
                           .time_since_epoch());
   }

   bool operator==(const DateHelper& other) const { return tp_ == other.tp_; }

   int64_t NanosSinceEpoch() const { return tp_.time_since_epoch().count(); }

   private:
   date::year_month_day YearMonthDay() const {
      return date::year_month_day{
         date::floor<date::days>(tp_)}; // NOLINT
   }

   date::time_of_day<std::chrono::nanoseconds> TimeOfDay() const {
      auto nanos_since_midnight =
         tp_ - date::floor<date::days>(tp_);
      return date::time_of_day<std::chrono::nanoseconds>(
         nanos_since_midnight);
   }

   std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp_;
};
//end adapted from apache gandiva

extern "C" uint64_t rt_extract_second(uint64_t nanos) {
   return DateHelper(nanos).TmSec();
}
extern "C" uint64_t rt_extract_minute(uint64_t nanos) {
   return DateHelper(nanos).TmMin();
}
extern "C" uint64_t rt_extract_hour(uint64_t nanos) {
   return DateHelper(nanos).TmHour();
}
extern "C" uint64_t rt_extract_dow(uint64_t nanos) {
   return DateHelper(nanos).TmWday() + 1;
}
extern "C" uint64_t rt_extract_day(uint64_t nanos) {
   return DateHelper(nanos).TmMday();
}
extern "C" uint64_t rt_extract_month(uint64_t nanos) {
   return DateHelper(nanos).TmMon() + 1;
}
extern "C" uint64_t rt_extract_year(uint64_t nanos) {
   return DateHelper(nanos).TmYear() + 1900;
}
extern "C" uint64_t rt_extract_doy(uint64_t nanos) {
   return DateHelper(nanos).TmYday();
}
extern "C" uint64_t rt_timestamp_add_months(int64_t nanos, uint32_t months) {
   return DateHelper(nanos).AddMonths(months).NanosSinceEpoch();
}
extern "C" uint64_t rt_timestamp_subtract_months(int64_t nanos, uint32_t months) {
   return DateHelper(nanos).AddMonths(-months).NanosSinceEpoch();
}