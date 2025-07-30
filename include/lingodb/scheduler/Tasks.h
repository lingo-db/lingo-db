
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
      context->setupPython();
   }
   void teardown() override {
      runtime::setCurrentExecutionContext(nullptr);
      context->teardownPython();
   }
};
class TaskWithImplicitContext : public Task {
   runtime::ExecutionContext* context;

   public:
   TaskWithImplicitContext() : context(runtime::getCurrentExecutionContext()) {}
   void setup() override {
      runtime::setCurrentExecutionContext(context);
      context->setupPython();
   }
   void teardown() override {
      runtime::setCurrentExecutionContext(nullptr);
      context->teardownPython();
   }
};
} // namespace lingodb::scheduler

#endif //LINGODB_SCHEDULER_TASKS_H
