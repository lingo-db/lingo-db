#include "arrow/vendored/datetime/date.h"
#include "runtime/DateRuntime.h"
#include "runtime/helpers.h"
//adapted from apache gandiva
//source: https://github.com/apache/arrow/blob/3da66003ab2543c231fdf6551c2eb886f9a7e68f/cpp/src/gandiva/precompiled/epoch_time_point.h
//Apache-2.0 License
namespace date = arrow_vendored::date;
class DateHelper {
   public:
   explicit DateHelper(std::chrono::nanoseconds nanosSinceEpoch)
      : tp(nanosSinceEpoch) {}

   explicit DateHelper(int64_t nanosecondsSinceEpoch)
      : DateHelper(std::chrono::nanoseconds(nanosecondsSinceEpoch)) {}

   int64_t tmYear() const { return static_cast<int>(yearMonthDay().year()) - 1900; }

   int64_t tmMon() const { return static_cast<unsigned int>(yearMonthDay().month()) - 1; }

   int64_t tmYday() const {
      auto toDays = date::floor<date::days>(tp);
      auto firstDayInYear = date::sys_days{
         yearMonthDay().year() / date::jan / 1};
      return (toDays - firstDayInYear).count();
   }

   int64_t tmMday() const { return static_cast<unsigned int>(yearMonthDay().day()); }

   DateHelper addMonths(int numMonths) const {
      auto ymd = yearMonthDay() + date::months(numMonths);
      return DateHelper((date::sys_days{ymd} + // NOLINT
                         timeOfDay().to_duration())
                           .time_since_epoch());
   }

   bool operator==(const DateHelper& other) const { return tp == other.tp; }

   int64_t nanosSinceEpoch() const { return tp.time_since_epoch().count(); }

   private:
   date::year_month_day yearMonthDay() const {
      return date::year_month_day{
         date::floor<date::days>(tp)}; // NOLINT
   }

   date::time_of_day<std::chrono::nanoseconds> timeOfDay() const {
      auto nanosSinceMidnight =
         tp - date::floor<date::days>(tp);
      return date::time_of_day<std::chrono::nanoseconds>(
         nanosSinceMidnight);
   }

   std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp;
};
//end adapted from apache gandiva
int64_t runtime::DateRuntime::subtractMonths(int64_t date, int64_t months) {
   return DateHelper(date).addMonths(-months).nanosSinceEpoch();
}
int64_t runtime::DateRuntime::addMonths(int64_t nanos, int64_t months) {
   return DateHelper(nanos).addMonths(months).nanosSinceEpoch();
}
int64_t runtime::DateRuntime::extractYear(int64_t date) {
   return DateHelper(date).tmYear() + 1900;
}
int64_t runtime::DateRuntime::extractMonth(int64_t date) {
   return DateHelper(date).tmMon() + 1;
}
int64_t runtime::DateRuntime::extractDay(int64_t date) {
   return DateHelper(date).tmMday();
}
