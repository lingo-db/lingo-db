
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
#ifdef USE_CPYTHON_RUNTIME
      context->setupPython();
#endif
   }
   void teardown() override {
      runtime::setCurrentExecutionContext(nullptr);
#ifdef USE_CPYTHON_RUNTIME
      context->teardownPython();
#endif
   }
};
class TaskWithImplicitContext : public Task {
   runtime::ExecutionContext* context;

   public:
   TaskWithImplicitContext() : context(runtime::getCurrentExecutionContext()) {}
   void setup() override {
      runtime::setCurrentExecutionContext(context);
#ifdef USE_CPYTHON_RUNTIME
      context->setupPython();
#endif
   }
   void teardown() override {
      runtime::setCurrentExecutionContext(nullptr);
#ifdef USE_CPYTHON_RUNTIME
      context->teardownPython();
#endif
   }
};
} // namespace lingodb::scheduler

#endif //LINGODB_SCHEDULER_TASKS_H
