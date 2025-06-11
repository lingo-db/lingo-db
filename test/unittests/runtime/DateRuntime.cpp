#include "catch2/catch_all.hpp"

#include <chrono>
#include <iostream>
#include <string>

#include "lingodb/runtime/DateRuntime.h"
#include "lingodb/scheduler/Tasks.h"

#include <arrow/vendored/datetime/date.h>

namespace lingodb::runtime {

namespace {
class MockTaskWithContext : public lingodb::scheduler::TaskWithContext {
   std::function<void()> job;

   public:
   MockTaskWithContext(lingodb::runtime::ExecutionContext* context, std::function<void()> job) : TaskWithContext(context), job(std::move(job)) {}
   bool allocateWork() override {
      if (workExhausted.exchange(true)) {
         return false;
      }
      return true;
   }
   void performWork() override {
      job();
   }
};
template <class F>
void withContext(const F& f) {
   auto scheduler = lingodb::scheduler::startScheduler();
   auto session = Session::createSession();
   ExecutionContext context(*session);
   lingodb::scheduler::awaitEntryTask(std::make_unique<MockTaskWithContext>(&context, [&]() {
      f();
   }));
}

bool comparator(const std::string& part,
                const int yearFrom, const unsigned monthFrom, const unsigned dayFrom, const int hourFrom, const int minuteFrom, const int secondFrom,
                const int yearTo, const int monthTo, const int dayTo, const int hourTo, const int minuteTo, const int secondTo) {
   using namespace std::chrono;

   VarLen32 partVarLen = VarLen32::fromString(part);

   auto tpFrom = sys_days{year{yearFrom} / month{monthFrom} / day{dayFrom}} + hours{hourFrom} + minutes{minuteFrom} + seconds{secondFrom};
   auto tpTo = sys_days{year{yearTo} / monthTo / dayTo} + hours{hourTo} + minutes{minuteTo} + seconds{secondTo};

   // Convert to nanoseconds since epoch
   auto nsFrom = duration_cast<nanoseconds>(tpFrom.time_since_epoch()).count();
   auto nsTo = duration_cast<nanoseconds>(tpTo.time_since_epoch()).count();

   return DateRuntime::dateTrunc(partVarLen, nsFrom) == nsTo;
}

} // namespace

TEST_CASE("DateTrunc:YearTruncation") {
   withContext([]() {
      REQUIRE(comparator("year", 2025, 5, 17, 0, 0, 0, 2025, 1, 1, 0, 0, 0));
      REQUIRE(comparator("year", 1999, 12, 31, 0, 0, 0, 1999, 1, 1, 0, 0, 0));
   });
}

TEST_CASE("DateTrunc:MonthTruncation") {
   withContext([]() {
      REQUIRE(comparator("month", 2025, 5, 17, 0, 0, 0, 2025, 5, 1, 0, 0, 0));
      REQUIRE(comparator("month", 2020, 2, 29, 0, 0, 0, 2020, 2, 1, 0, 0, 0)); // Leap year
   });
}

TEST_CASE("DateTrunc:DayTruncation") {
   withContext([]() {
      REQUIRE(comparator("day", 2025, 5, 17, 0, 0, 0, 2025, 5, 17, 0, 0, 0)); // identity
   });
}

TEST_CASE("DateTrunc:AllMonths") {
   withContext([]() {
      for (int month = 1; month <= 12; ++month) {
         REQUIRE(comparator("month", 2021, month, 15, 0, 0, 0, 2021, month, 1, 0, 0, 0));
      }
   });
}

TEST_CASE("DateTrunc:HourTruncation") {
   withContext([]() {
      REQUIRE(comparator("hour", 2025, 6, 26, 14, 53, 22, 2025, 6, 26, 14, 0, 0));
      REQUIRE(comparator("hour", 2025, 1, 1, 0, 1, 1, 2025, 1, 1, 0, 0, 0));
      REQUIRE(comparator("hour", 1999, 12, 31, 23, 59, 59, 1999, 12, 31, 23, 0, 0));
   });
}

TEST_CASE("DateTrunc:MinuteTruncation") {
   withContext([]() {
      REQUIRE(comparator("minute", 2025, 6, 26, 14, 53, 22, 2025, 6, 26, 14, 53, 0));
      REQUIRE(comparator("minute", 2025, 1, 1, 0, 0, 59, 2025, 1, 1, 0, 0, 0));
      REQUIRE(comparator("minute", 1999, 12, 31, 23, 59, 1, 1999, 12, 31, 23, 59, 0));
   });
}

TEST_CASE("DateTrunc:SecondTruncation") {
   withContext([]() {
      REQUIRE(comparator("second", 2025, 6, 26, 14, 53, 22, 2025, 6, 26, 14, 53, 22));
      REQUIRE(comparator("second", 2025, 6, 26, 14, 53, 22, 2025, 6, 26, 14, 53, 22)); // idempotent
   });
}

TEST_CASE("DateTrunc:AllHours") {
   withContext([]() {
      for (int hour = 0; hour < 24; ++hour) {
         REQUIRE(comparator("hour", 2025, 6, 26, hour, 30, 45, 2025, 6, 26, hour, 0, 0));
      }
   });
}

TEST_CASE("DateTrunc:AllMinutes") {
   withContext([]() {
      for (int minute = 0; minute < 60; ++minute) {
         REQUIRE(comparator("minute", 2025, 6, 26, 14, minute, 59, 2025, 6, 26, 14, minute, 0));
      }
   });
}
} // namespace lingodb::runtime
