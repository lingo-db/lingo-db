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

int64_t toDateLike(const int yearIn, const unsigned monthIn = 1, const unsigned dayIn = 1, const int hourIn = 0, const int minuteIn = 0, const int secondIn = 0) {
   using namespace std::chrono;
   const auto tp = sys_days{year{yearIn} / month{monthIn} / day{dayIn}} + hours{hourIn} + minutes{minuteIn} + seconds{secondIn};
   const auto ns = duration_cast<nanoseconds>(tp.time_since_epoch()).count();
   return ns;
}

} // namespace

TEST_CASE("DateTrunc:YearTruncation") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("year");
      auto date1 = toDateLike(2025, 5, 17);
      auto date2 = toDateLike(2025);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, date1) == date2);

      auto date3 = toDateLike(1999, 12, 31);
      auto date4 = toDateLike(1999);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, date3) == date4);

   });
}

TEST_CASE("DateTrunc:MonthTruncation") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("month");
      auto date1 = toDateLike(2025, 5, 17);
      auto date2 = toDateLike(2025, 5, 1);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, date1) == date2);

      auto date3 = toDateLike(2020, 2, 29);
      auto date4 = toDateLike(2020, 2, 1);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, date3) == date4);
   });
}

TEST_CASE("DateTrunc:DayTruncation") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("day");
      auto date = toDateLike(2025, 5, 17);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, date) == date); // identity
   });
}

TEST_CASE("DateTrunc:AllMonths") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("month");
      for (int month = 1; month <= 12; ++month) {
         auto date = toDateLike(2021, month, 15);
         auto truncated = toDateLike(2021, month, 1);
         REQUIRE(DateRuntime::dateTrunc(partVarLen, date) == truncated);
      }
   });
}

TEST_CASE("DateTrunc:HourTruncation") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("hour");

      auto d1 = toDateLike(2025, 6, 26, 14, 53, 22);
      auto d2 = toDateLike(2025, 6, 26, 14);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, d1) == d2);

      auto d3 = toDateLike(2025, 1, 1, 0, 1, 1);
      auto d4 = toDateLike(2025, 1, 1, 0);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, d3) == d4);

      auto d5 = toDateLike(1999, 12, 31, 23, 59, 59);
      auto d6 = toDateLike(1999, 12, 31, 23);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, d5) == d6);
   });
}

TEST_CASE("DateTrunc:MinuteTruncation") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("minute");

      auto d1 = toDateLike(2025, 6, 26, 14, 53, 22);
      auto d2 = toDateLike(2025, 6, 26, 14, 53);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, d1) == d2);

      auto d3 = toDateLike(2025, 1, 1, 0, 0, 59);
      auto d4 = toDateLike(2025, 1, 1, 0, 0);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, d3) == d4);

      auto d5 = toDateLike(1999, 12, 31, 23, 59, 1);
      auto d6 = toDateLike(1999, 12, 31, 23, 59);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, d5) == d6);
   });
}

TEST_CASE("DateTrunc:SecondTruncation") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("second");

      auto d1 = toDateLike(2025, 6, 26, 14, 53, 22);
      REQUIRE(DateRuntime::dateTrunc(partVarLen, d1) == d1); // identity

      REQUIRE(DateRuntime::dateTrunc(partVarLen, DateRuntime::dateTrunc(partVarLen, d1)) == d1); // idempotent
   });
}

TEST_CASE("DateTrunc:AllHours") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("hour");

      for (int hour = 0; hour < 24; ++hour) {
         auto date = toDateLike(2025, 6, 26, hour, 30, 45);
         auto truncated = toDateLike(2025, 6, 26, hour);
         REQUIRE(DateRuntime::dateTrunc(partVarLen, date) == truncated);
      }
   });
}

TEST_CASE("DateTrunc:AllMinutes") {
   withContext([]() {
      VarLen32 partVarLen = VarLen32::fromString("minute");

      for (int minute = 0; minute < 60; ++minute) {
         REQUIRE(comparator("minute", 2025, 6, 26, 14, minute, 59, 2025, 6, 26, 14, minute, 0));
      }
         auto date = toDateLike(2025, 6, 26, 14, minute, 59);
         auto truncated = toDateLike(2025, 6, 26, 14, minute);
         REQUIRE(DateRuntime::dateTrunc(partVarLen, date) == truncated);
      }
   });
}

   });
}
} // namespace lingodb::runtime
