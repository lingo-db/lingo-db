#ifndef LINGODB_SCHEDULER_SCHEDULER_H
#define LINGODB_SCHEDULER_SCHEDULER_H
#include "lingodb/scheduler/Task.h"
#include <memory>
namespace lingodb::scheduler {

//handle to a scheduler. When the last handle is destroyed, the scheduler is stopped (waited for) and deleted.
class SchedulerHandle {
   public:
   SchedulerHandle();
   ~SchedulerHandle();
};

//starts a scheduler with a given number of workers.
//If numWorkers is 0, the number of workers is determined by the LINGODB_PARALLELISM environment variable or std::thread::hardware_concurrency()
//If initialFiberAllocs is not 0, than number of initialFiberAllocs fiber will be allocated initially.
//if a scheduler is already running, a handle to this scheduler is returned (the number of workers is ignored)
std::unique_ptr<SchedulerHandle> startScheduler(size_t numWorkers = 0);
//waits for the scheduler to finish the current task (this is a blocking call, designed for calling from a non-worker thread)
void awaitEntryTask(std::unique_ptr<Task> task);
//waits for the scheduler to finish the current task (this will yield the current worker thread, only use from a worker thread)
void awaitChildTask(std::unique_ptr<Task> task);

//returns the number of workers in the scheduler
size_t getNumWorkers();
//returns the id of the current worker thread
size_t currentWorkerId();
#if !ASAN_ACTIVE
uint8_t* getStackBoundary();
#endif

} // namespace lingodb::scheduler

#endif //LINGODB_SCHEDULER_SCHEDULER_H
