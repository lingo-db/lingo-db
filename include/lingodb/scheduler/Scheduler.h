
#ifndef LINGODB_SCHEDULER_SCHEDULER_H
#define LINGODB_SCHEDULER_SCHEDULER_H
#include "lingodb/scheduler/Task.h"
#include <functional>
#include <memory>
#include <thread>
namespace lingodb::scheduler {
class Scheduler {
   protected:
   size_t numWorkers;

   public:
   Scheduler(size_t numWorkers = std::thread::hardware_concurrency()) : numWorkers(numWorkers) {
   }
   virtual void start() = 0;
   virtual void enqueueTask(std::unique_ptr<Task>&& task) = 0;
   virtual void join() = 0;
   virtual ~Scheduler() {
   }
   size_t getNumWorkers() {
      return numWorkers;
   }
};

class SchedulerHandle {
   public:
   SchedulerHandle();
   ~SchedulerHandle();
};
std::unique_ptr<SchedulerHandle> startScheduler(size_t numWorkers = 0);
void awaitEntryTask(std::unique_ptr<Task> task);
void awaitChildTask(std::unique_ptr<Task> task);

size_t getNumWorkers();
size_t currentWorkerId();

} // namespace lingodb::scheduler

#endif //LINGODB_SCHEDULER_SCHEDULER_H
