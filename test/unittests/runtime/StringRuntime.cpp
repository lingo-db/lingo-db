#include "catch2/catch_all.hpp"

#include <string>

#include "lingodb/runtime/StringRuntime.h"

#include <iostream>
#include <lingodb/scheduler/Tasks.h>

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

bool comparator(const std::string& input, const std::string& pattern, const std::string& replace, const std::string& expected) {
   // Convert to VarLen
   const VarLen32 inputVarLen32 = VarLen32::fromString(input);
   const VarLen32 patternVarLen32 = VarLen32::fromString(pattern);
   const VarLen32 replaceVarLen32 = VarLen32::fromString(replace);
   const VarLen32 expectedVarLen32 = VarLen32::fromString(expected);

   const VarLen32 result = StringRuntime::regexpReplace(inputVarLen32, patternVarLen32, replaceVarLen32);
   return StringRuntime::compareEq(result, expectedVarLen32);
}

} // namespace

TEST_CASE("RegexpReplace:BasicSubstitution") {
   withContext([]() {
      REQUIRE(comparator("Hello World", "World", "Universe", "Hello Universe"));
   });
}

TEST_CASE("RegexpReplace:GlobalSubstitution") {
   withContext([]() {
      REQUIRE(comparator("abc abc abc", "abc", "xyz", "xyz xyz xyz"));
   });
}

TEST_CASE("RegexpReplace:EmptyReplacement") {
   withContext([]() {
      REQUIRE(comparator("banana", "a", "", "bnn"));
   });
}

TEST_CASE("RegexpReplace:SpecialCharacters") {
   withContext([]() {
      REQUIRE(comparator("a1b2c3", "[0-9]", "X", "aXbXcX"));
   });
}

TEST_CASE("RegexpReplace:NoMatch") {
   withContext([]() {
      REQUIRE(comparator("hello", "z", "X", "hello"));
   });
}

TEST_CASE("RegexpReplace:StartEndAnchors") {
   withContext([]() {
      REQUIRE(comparator("foobar", "^foo", "baz", "bazbar"));
   });
}

TEST_CASE("RegexpReplace:EscapedCharacters") {
   withContext([]() {
      REQUIRE(comparator("a+b+c+", "\\+", "-", "a-b-c-"));
   });
}
} // namespace lingodb::runtime
