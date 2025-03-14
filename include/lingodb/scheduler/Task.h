#ifndef LINGODB_SCHEDULER_TASK_H
#define LINGODB_SCHEDULER_TASK_H
#include <atomic>
#include <limits>

namespace lingodb::scheduler {
class Task {
   protected:
   std::atomic<bool> workExhausted{false};

   public:
   bool hasWork() {
      return !workExhausted.load();
   }

   virtual bool allocateWork() = 0;
   virtual void performWork() = 0;
   //e.g., to prepare environment
   virtual void setup() {}
   virtual void teardown() {}
   virtual ~Task() {}
};
} // namespace lingodb::scheduler
#endif //LINGODB_SCHEDULER_TASK_H
