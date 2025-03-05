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

   virtual size_t workAmount() { return std::numeric_limits<size_t>::max(); }
   virtual bool reserveWork() = 0;
   virtual void consumeWork() = 0;
   virtual ~Task() {}
};
} // namespace lingodb::scheduler
#endif //LINGODB_SCHEDULER_TASK_H
