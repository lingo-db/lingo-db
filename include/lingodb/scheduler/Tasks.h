
#ifndef LINGODB_SCHEDULER_TASKS_H
#define LINGODB_SCHEDULER_TASKS_H
#include "lingodb/runtime/ExecutionContext.h"
namespace lingodb::scheduler {
class TaskWithContext : public Task {
   runtime::ExecutionContext* context;

   public:
   TaskWithContext(runtime::ExecutionContext* context) : context(context) {}
   void setup() override {
      runtime::setCurrentExecutionContext(context);
   }
   void teardown() override {
      runtime::setCurrentExecutionContext(nullptr);
   }
};
class TaskWithImplicitContext : public Task {
   runtime::ExecutionContext* context;

   public:
   TaskWithImplicitContext() : context(runtime::getCurrentExecutionContext()) {}
   void setup() override {
      runtime::setCurrentExecutionContext(context);
   }
   void teardown() override {
      runtime::setCurrentExecutionContext(nullptr);
   }
};
class SimpleTask : public lingodb::scheduler::Task {
   std::function<void()> func;

   public:
   SimpleTask(std::function<void()> func) : func(func) {}

   virtual bool allocateWork() override {
      if (workExhausted.exchange(true)) {
         return false;
      }
      return true;
   }
   virtual void performWork() override {
      func();
   }
};

} // namespace lingodb::scheduler

#endif //LINGODB_SCHEDULER_TASKS_H
