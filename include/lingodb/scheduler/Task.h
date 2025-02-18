#ifndef LINGODB_SCHEDULER_TASK_H
#define LINGODB_SCHEDULER_TASK_H
#include <atomic>
namespace lingodb::scheduler {
class Task {
   protected:
   bool singleRun{false};
   std::atomic<bool> workExhausted{false};

   public:
   bool onlySingleRun() {
      return singleRun;
   }
   bool hasWork() {
      return !workExhausted.load();
   }

   virtual void run() = 0;
   virtual ~Task() {}
};
} // namespace lingodb::scheduler
#endif //LINGODB_SCHEDULER_TASK_H
