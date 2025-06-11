#include "lingodb/runtime/DateRuntime.h"
#include "arrow/vendored/datetime/date.h"
#include "lingodb/runtime/StringRuntime.h"
#include "lingodb/runtime/helpers.h"

#include <chrono>
//adapted from apache gandiva
//source: https://github.com/apache/arrow/blob/3da66003ab2543c231fdf6551c2eb886f9a7e68f/cpp/src/gandiva/precompiled/epoch_time_point.h
//Apache-2.0 License
namespace date = arrow_vendored::date;
class DateHelper {
   public:
   // Note: this limits the number of years into the future/past that we can represent
   explicit DateHelper(std::chrono::nanoseconds nanosSinceEpoch)
      : tp(nanosSinceEpoch) {}

   explicit DateHelper(int64_t nanosecondsSinceEpoch)
      : DateHelper(std::chrono::nanoseconds(nanosecondsSinceEpoch)) {}

   int64_t tmYear() const { return static_cast<int>(yearMonthDay().year()) - 1900; }
   int64_t tmMon() const { return static_cast<unsigned int>(yearMonthDay().month()) - 1; }
   int64_t tmHour() const { return timeOfDay().hours().count(); }
   int64_t tmMinute() const { return timeOfDay().minutes().count(); }
   int64_t tmSecond() const { return timeOfDay().seconds().count(); }

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

namespace {
int64_t toEpochNs(int year, unsigned month, unsigned day, int64_t hour, int64_t minute, int64_t second) {
   using namespace std::chrono;
   using namespace date;

   // Construct the time point from YMDHMS
   auto ymd = std::chrono::year_month_day{std::chrono::year{year}, std::chrono::month{month}, std::chrono::day{day}};
   std::chrono::sys_days dayPoint = std::chrono::sys_days{ymd};
   auto tp = dayPoint + hours{hour} + minutes{minute} + seconds{second};

   return duration_cast<nanoseconds>(tp.time_since_epoch()).count();
}
} // namespace

int64_t lingodb::runtime::DateRuntime::subtractMonths(int64_t date, int64_t months) {
   return DateHelper(date).addMonths(-months).nanosSinceEpoch();
}
int64_t lingodb::runtime::DateRuntime::addMonths(int64_t nanos, int64_t months) {
   return DateHelper(nanos).addMonths(months).nanosSinceEpoch();
}
int64_t lingodb::runtime::DateRuntime::dateDiffSeconds(int64_t start, int64_t end) {
   auto diffNanos = end - start;
   return diffNanos / (1000000000ull);
}
int64_t lingodb::runtime::DateRuntime::extractYear(int64_t date) {
   return DateHelper(date).tmYear() + 1900;
}
int64_t lingodb::runtime::DateRuntime::extractMonth(int64_t date) {
   return DateHelper(date).tmMon() + 1;
}
int64_t lingodb::runtime::DateRuntime::extractDay(int64_t date) {
   return DateHelper(date).tmMday();
}
int64_t lingodb::runtime::DateRuntime::extractHour(int64_t date) {
   return DateHelper(date).tmHour();
}
int64_t lingodb::runtime::DateRuntime::extractMinute(int64_t date) {
   return DateHelper(date).tmMinute();
}
int64_t lingodb::runtime::DateRuntime::extractSecond(int64_t date) {
   return DateHelper(date).tmSecond();
}
int64_t lingodb::runtime::DateRuntime::dateTrunc(VarLen32 part, int64_t date) {
   // TODO we only implement a subset of the postgres functionality
   // https://www.postgresql.org/docs/current/functions-datetime.html#FUNCTIONS-DATETIME-TRUNC

   const int year = extractYear(date);
   unsigned month = 1;
   unsigned day = 1;
   int64_t hour = 0;
   int64_t minute = 0;
   int64_t second = 0;

   if (StringRuntime::len(part) == 4 && // String length must be checked before comparison
       StringRuntime::compareEq(part, VarLen32::fromString("year"))) {
      return toEpochNs(year, month, day, hour, minute, second);
   }

   month = extractMonth(date);
   if (StringRuntime::len(part) == 5 &&
       StringRuntime::compareEq(part, VarLen32::fromString("month"))) {
      return toEpochNs(year, month, day, hour, minute, second);
   }

   day = extractDay(date);
   if (StringRuntime::len(part) == 3 &&
       StringRuntime::compareEq(part, VarLen32::fromString("day"))) {
      return toEpochNs(year, month, day, hour, minute, second);
   }

   hour = extractHour(date);
   if (StringRuntime::len(part) == 4 &&
       StringRuntime::compareEq(part, VarLen32::fromString("hour"))) {
      return toEpochNs(year, month, day, hour, minute, second);
   }

   minute = extractMinute(date);
   if (StringRuntime::len(part) == 6 &&
       StringRuntime::compareEq(part, VarLen32::fromString("minute"))) {
      return toEpochNs(year, month, day, hour, minute, second);
   }

   second = extractSecond(date);
   if (StringRuntime::len(part) == 6 &&
       StringRuntime::compareEq(part, VarLen32::fromString("second"))) {
      return toEpochNs(year, month, day, hour, minute, second);
   }

   throw std::runtime_error("Invalid or unimplemented date part");
}
