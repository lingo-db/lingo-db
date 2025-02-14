#include "lingodb/utility/Tracer.h"
#include <atomic>
#include <boost/context/fiber.hpp>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <memory>
#include <thread>

#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Task.h"

namespace lingodb::scheduler {
class Worker;

struct TaskWrapper;

class Fiber {
   static constexpr size_t stackSize = 1 << 20;

   struct LocalAllocator {
      Fiber* fiber = nullptr;

      boost::context::stack_context allocate() {
         assert(!fiber->stackAllocated);
         boost::context::stack_context sctx;
         sctx.size = stackSize;
         sctx.sp = fiber->stackSpace + stackSize;
         fiber->stackAllocated = true;
         return sctx;
      }

      void deallocate(boost::context::stack_context&) noexcept {
         assert(fiber->stackAllocated);
         fiber->stackAllocated = false;
      }

      friend class Fiber;
   };

   alignas(64) std::byte stackSpace[stackSize];
   bool stackAllocated = false;
   bool isRunning = false;
   bool done = true;
   boost::context::fiber fiber;
   boost::context::fiber sink;
   Worker* worker = nullptr;
   TaskWrapper* task = nullptr;

   public:
   bool run(Worker* worker, TaskWrapper* taskWrapper, const std::function<void()>&& f) {
      this->worker = worker;
      this->task = taskWrapper;
      done = false;
      isRunning = true;
      fiber = boost::context::fiber{std::allocator_arg, LocalAllocator{this}, [&](boost::context::fiber&& boostSink) {
                                       sink = std::move(boostSink);
                                       f();
                                       isRunning = false;
                                       done = true;
                                       return std::move(sink);
                                    }}
                 .resume();
      return done;
   }

   Worker* getWorker() {
      return worker;
   }

   TaskWrapper* getTask() {
      return task;
   }

   bool resume() {
      assert(!isRunning);
      assert(!done);
      isRunning = true;
      fiber = std::move(fiber).resume();
      return done;
   }

   void yield() {
      assert(isRunning);
      assert(!done);
      isRunning = false;
      sink = std::move(sink).resume();
   }

   Fiber() = default;

   ~Fiber() {
      assert(done);
   }
};

struct TaskWrapper {
   std::unique_ptr<Task> task;

   std::atomic<bool> coolingDown = false;
   bool finalized = false;
   TaskWrapper* next = nullptr;
   TaskWrapper* prev = nullptr;
   std::atomic<int64_t> yieldedFibers = 0;
   std::atomic<int64_t> nonCompletedFibers = 0;
   std::atomic<int64_t> deployedOnWorkers = 0;
   std::unique_ptr<Fiber> waitingOnTaskCompletion = nullptr;
   std::function<void()> beforeDestroyFn = nullptr;

   void finalize();

   bool done() {
      return !task->hasWork() && nonCompletedFibers.load() == 0;
   }

   bool startFiber() {
      if (task->hasWork()) {
         nonCompletedFibers++;
         return true;
      }
      return false;
   }

   bool finishFiber() {
      if (nonCompletedFibers.fetch_sub(1) == 1 && !task->hasWork()) {
         return true;
      }
      return false;
   }

   void yieldFiber() {
      yieldedFibers++;
   }

   void unYieldFiber() {
      yieldedFibers--;
   }
};

class Scheduler {
   size_t numWorkers;

   std::atomic<bool> shutdown{false};
   std::vector<std::thread> workerThreads;
   Worker* idleWorkers = nullptr;
   std::mutex taskQueueMutex;
   std::mutex taskReturnMutex;
   TaskWrapper* taskHead = nullptr;
   TaskWrapper* taskTail = nullptr;

   //queue for tasks that are cooling down i.e. they don't have work left to do, but are also not yet finished.
   std::mutex coolingDownMutex;
   TaskWrapper* coolingDownHead = nullptr;
   TaskWrapper* coolingDownTail = nullptr;

   public:
   Scheduler(size_t numWorkers = std::thread::hardware_concurrency()) : numWorkers(numWorkers) {
   }
   size_t getNumWorkers() {
      return numWorkers;
   }

   void putWorkerToSleep(Worker* worker);

   void start();

   void stop();

   void join() {
      for (auto& workerThread : workerThreads) {
         workerThread.join();
      }
   }

   bool isShutdown() {
      return shutdown.load();
   }

   //insert task into "active task queue"
   void enqueueTask(TaskWrapper* wrapper);

   void enqueueTask(std::unique_ptr<Task>&& task) {
      enqueueTask(new TaskWrapper{std::move(task)});
   }

   //remove task from "active task queue"
   void dequeueTaskLocked(TaskWrapper* task) {
      if (task->prev) {
         task->prev->next = task->next;
         task->prev = nullptr;
      } else {
         taskHead = task->next;
      }
      if (task->next) {
         task->next->prev = task->prev;
         task->next = nullptr;
      } else {
         taskTail = task->prev;
      }
   }

   TaskWrapper* getTask() {
      std::lock_guard<std::mutex> lock(taskQueueMutex);
      auto* potentialTask = taskHead;
      size_t minYieldCount = std::numeric_limits<size_t>::max();
      TaskWrapper* minYieldTask = nullptr;
      while (potentialTask) {
         //if we see a task that has no work left, we move it to the cooling down queue
         if (!potentialTask->task->hasWork()) {
            auto* toCoolDown = potentialTask;
            potentialTask = potentialTask->next;
            dequeueTaskLocked(toCoolDown);
            std::lock_guard<std::mutex> lock1(coolingDownMutex);
            if (coolingDownTail) {
               coolingDownTail->next = toCoolDown;
               toCoolDown->prev = coolingDownTail;
               coolingDownTail = toCoolDown;
            } else {
               coolingDownHead = toCoolDown;
               coolingDownTail = toCoolDown;
            }
            toCoolDown->coolingDown = true;
            continue;
         }
         auto yieldedFibers = potentialTask->yieldedFibers.load();
         if (yieldedFibers == 0) {
            potentialTask->deployedOnWorkers++;
            return potentialTask;
         } else {
            if (yieldedFibers < 0 || size_t(yieldedFibers) < minYieldCount) {
               minYieldCount = yieldedFibers;
               minYieldTask = potentialTask;
            }
         }
         potentialTask = potentialTask->next;
      }
      if (minYieldTask) {
         minYieldTask->deployedOnWorkers++;
      }
      return minYieldTask;
   }

   void returnTask(TaskWrapper* task) {
      // Multiple worker may call here concurrently. Avoid one worker already called `delete task`
      // but another worker is still at `task->finalized`
      std::lock_guard<std::mutex> lock(taskReturnMutex);
      auto deployedNum = task->deployedOnWorkers.fetch_sub(1);
      if (task->finalized) {
         if (deployedNum == 1) {
            if (task->beforeDestroyFn) {
               task->beforeDestroyFn();
            }
            delete task;
         }
      }
   }

   void finalizeTask(TaskWrapper* task) {
      if (task->coolingDown) {
         //simple case: already in cooling down queue
         // -> only need to lock cooling down queue
         std::lock_guard<std::mutex> lock(coolingDownMutex);
         if (task->prev) {
            task->prev->next = task->next;
         } else {
            coolingDownHead = task->next;
         }
         if (task->next) {
            task->next->prev = task->prev;
         } else {
            coolingDownTail = task->prev;
         }
      } else {
         //not in cooling down queue
         // -> need to lock both task queue and cooling down queue
         bool alreadyCoolingDown;
         {
            std::lock_guard<std::mutex> lock(taskQueueMutex);
            alreadyCoolingDown = task->coolingDown;
            if (!alreadyCoolingDown) {
               dequeueTaskLocked(task);
            }
         }
         // - If alreadyCoolingDown is true, last attempt to dequeue task is failed. task is appedn to coolingDown queue. need to remove it.
         // - If alreadyCoolingDown is false, last attempt to dequeue task is ok. task is not inserted to coolingDown queue. no need to remove it.
         if (alreadyCoolingDown) {
            std::lock_guard<std::mutex> lock(coolingDownMutex);
            if (task->prev) {
               task->prev->next = task->next;
            } else {
               coolingDownHead = task->next;
            }
            if (task->next) {
               task->next->prev = task->prev;
            } else {
               coolingDownTail = task->prev;
            }
         }
      }
      task->finalize();
      task->finalized = true;
   }
};

class Worker {
   class FiberAllocator {
      size_t numAllocated;
      size_t maxFibers;
      std::deque<std::unique_ptr<Fiber>> allocatedAndAvailableFibers;

      public:
      explicit FiberAllocator(size_t maxFibers) : numAllocated(0), maxFibers(maxFibers) {
      }

      bool canAllocate() {
         return numAllocated < maxFibers || !allocatedAndAvailableFibers.empty();
      }

      std::unique_ptr<Fiber> allocate() {
         if (allocatedAndAvailableFibers.empty()) {
            if (numAllocated < maxFibers) {
               numAllocated++;
               return std::make_unique<Fiber>();
            } else {
               return {};
            }
         } else {
            auto fiber = std::move(allocatedAndAvailableFibers.back());
            allocatedAndAvailableFibers.pop_back();
            return fiber;
         }
      }

      void deallocate(std::unique_ptr<Fiber>&& fiber) {
         allocatedAndAvailableFibers.push_back(std::move(fiber));
      }
   };

   Scheduler& scheduler;

   //for managing a pool of fibers/stacks
   FiberAllocator fiberAllocator;

   //all fibers that are currently runnable
   std::mutex fiberMutex;
   std::deque<std::unique_ptr<Fiber>> runnableFibers;

   std::unique_ptr<Fiber> currentFiber;

   TaskWrapper* currentTask = nullptr;

   public:
   //for cheaply collecting idle workers
   Worker* nextIdleWorker = nullptr;
   bool isInIdleList = false;
   // for sleeping / waking up if there is nothing to do
   std::mutex mutex;
   std::condition_variable cv;
   size_t workerId;
   bool allowedToSleep = true;

   Worker(Scheduler& scheduler, size_t id) : scheduler(scheduler), fiberAllocator(64), workerId(id) {
   }

   void wakeupFiber(std::unique_ptr<Fiber>&& fiber) {
      {
         std::lock_guard<std::mutex> fiberQueueLock(fiberMutex);
         runnableFibers.push_back(std::move(fiber));
      }
      {
         std::unique_lock<std::mutex> lock(mutex);
         allowedToSleep = false;
         cv.notify_one();
      }
   }

   void awaitChildTask(std::unique_ptr<Task> task) {
      TaskWrapper* taskWrapper = new TaskWrapper{std::move(task)};
      Fiber& fiber = *currentFiber;
      taskWrapper->waitingOnTaskCompletion = std::move(currentFiber);
      scheduler.enqueueTask(taskWrapper);

      fiber.yield();
   }

   void work() {
      while (!scheduler.isShutdown()) {
         {
            std::lock_guard<std::mutex> lock(fiberMutex);

            if (!runnableFibers.empty()) {
               currentFiber = std::move(runnableFibers.front());
               runnableFibers.pop_front();
            }
         }
         auto handleFiberComplete = [&]() {
            auto* task = currentFiber->getTask();
            if (task) {
               if (task->finishFiber()) {
                  scheduler.finalizeTask(task);
               }
            }
            fiberAllocator.deallocate(std::move(currentFiber));
         };
         if (currentFiber) {
            if (currentFiber->resume()) {
               //unyield because it was previously registered as yielded
               if (currentFiber->getTask()) {
                  currentFiber->getTask()->unYieldFiber();
               }
               auto* resumeTask = currentFiber->getTask();
               handleFiberComplete();
               scheduler.returnTask(resumeTask);
            }
            assert(!currentFiber);
         }
         if (fiberAllocator.canAllocate()) {
            TaskWrapper* currTask = nullptr;
            {
               std::unique_lock<std::mutex> lock(mutex);
               if (this->currentTask) {
                  currTask = this->currentTask;
                  this->currentTask = nullptr;
               }
            }
            if (currTask && !currTask->task->hasWork()) {
               scheduler.returnTask(currTask);
               currTask = nullptr;
               continue;
            }
            if (!currTask) {
               currTask = scheduler.getTask();
            }

            if (currTask) {
               if (!currTask->startFiber()) {
                  scheduler.returnTask(currTask);
                  continue;
               }
               // if (currTask && currTask->startFiber()) {
               //work on (part of) (new) task
               currentFiber = fiberAllocator.allocate();
               assert(currentFiber);
               auto fiberDone = currentFiber->run(this, currTask, [&] {
                  currTask->task->run();
               });
               if (fiberDone) {
                  handleFiberComplete();
               } else {
                  currTask->yieldFiber();
                  // yielded fiber does't finished. continue and not execute following cleanup logic.
                  continue;
               }
               if (currTask->task->hasWork()) {
                  std::lock_guard<std::mutex> lock(mutex);
                  this->currentTask = currTask;
               } else {
                  scheduler.returnTask(currTask);
               }
               //assert(!currentFiber);

               continue;
            } else {
               scheduler.putWorkerToSleep(this);
            }

         } else {
            // we have a lot of fibers -> so don't sleep :)
            continue;
         }
      }
   }
};

namespace {
Scheduler* scheduler;
static thread_local Worker* currentWorker;
static std::atomic<size_t> numSchedulerUsers = 0;

void stopCurrentScheduler() {
   if (scheduler) {
      scheduler->stop();
   }
}
} // end namespace

void Scheduler::start() {
   scheduler = this;
   for (size_t i = 0; i < numWorkers; i++) {
      workerThreads.emplace_back([this, i] {
         Worker worker(*this, i);
         currentWorker = &worker;
         worker.work();
         currentWorker = nullptr;
      });
   }
}

void Scheduler::stop() {
   shutdown.store(true);
   std::unique_lock<std::mutex> lock(taskQueueMutex);
   size_t cntr = 0;
   while (idleWorkers) {
      assert(cntr++ < numWorkers);
      auto* currWorker = idleWorkers;
      currWorker->isInIdleList = false;
      idleWorkers = currWorker->nextIdleWorker;
      currWorker->cv.notify_one();
   }
}

void Scheduler::enqueueTask(TaskWrapper* wrapper) {
   {
      std::lock_guard<std::mutex> lock(taskQueueMutex);
      if (taskTail) {
         taskTail->next = wrapper;
         wrapper->prev = taskTail;
         taskTail = wrapper;
      } else {
         taskHead = wrapper;
         taskTail = wrapper;
      }
   }
   std::lock_guard<std::mutex> lock(taskQueueMutex);
   size_t cntr = 0;
   while (idleWorkers) {
      assert(cntr++ < numWorkers);
      Worker* currWorker = idleWorkers;
      std::unique_lock<std::mutex> workerLock(currWorker->mutex);
      currWorker->isInIdleList = false;
      idleWorkers = idleWorkers->nextIdleWorker;

      currWorker->cv.notify_one();
   }
}

void Scheduler::putWorkerToSleep(Worker* worker) {
   if (isShutdown()) {
      return;
   }
   std::unique_lock<std::mutex> lock(taskQueueMutex);
   std::unique_lock<std::mutex> workerLock(worker->mutex);
   if (worker->allowedToSleep) {
      if (!worker->isInIdleList) {
         worker->nextIdleWorker = idleWorkers;
         idleWorkers = worker;
         worker->isInIdleList = true;
      }
      lock.unlock();
      worker->cv.wait(workerLock);
   } else {
      worker->allowedToSleep = true;
   }
}

void TaskWrapper::finalize() {
   if (waitingOnTaskCompletion) {
      auto* worker = waitingOnTaskCompletion->getWorker();
      assert(worker);
      worker->wakeupFiber(std::move(waitingOnTaskCompletion));
   }
}

void awaitEntryTask(std::unique_ptr<Task> task) {
   TaskWrapper* taskWrapper = new TaskWrapper{std::move(task)};
   std::condition_variable finished;
   std::mutex mutex;
   taskWrapper->beforeDestroyFn = [&]() {
      finished.notify_one();
   };
   scheduler->enqueueTask(taskWrapper);
   std::unique_lock<std::mutex> lk(mutex);
   finished.wait(lk);
}
void awaitChildTask(std::unique_ptr<Task> task) {
   currentWorker->awaitChildTask(std::move(task));
}

std::unique_ptr<SchedulerHandle> startScheduler(size_t numWorkers) {
   if (!scheduler) {
      // two ways of setting number of workers.
      // - if provided actual param for createScheduler, this will override LINGODB_PARALLELISM system var
      // - if no actual param provided, LINGODB_PARALLELISM is used
      // - if no actual param and no LINGODB_PARALLELISM provided, hardware_concurrency() is used
      if (numWorkers == 0) {
         // LINGODB_PARALLELISM is one of ["OFF", positive integer]
         numWorkers = std::thread::hardware_concurrency();
         if (const char* mode = std::getenv("LINGODB_PARALLELISM")) {
            if (std::string(mode) == "OFF") {
               numWorkers = 1;
            } else if (std::stol(mode) > 0) {
               numWorkers = std::stol(mode);
            }
         }
      }
      scheduler = new Scheduler(numWorkers);
      scheduler->start();
   }
   return std::make_unique<SchedulerHandle>();
}

size_t getNumWorkers() {
   if (scheduler == nullptr) {
      assert(false);
      return 0;
   }
   return scheduler->getNumWorkers();
}
size_t currentWorkerId() {
   if (currentWorker) {
      return currentWorker->workerId;
   }
   assert(false);
   return std::numeric_limits<size_t>::max();
}
} // namespace lingodb::scheduler

lingodb::scheduler::SchedulerHandle::SchedulerHandle() {
   numSchedulerUsers++;
}
lingodb::scheduler::SchedulerHandle::~SchedulerHandle() {
   if (numSchedulerUsers.fetch_sub(1) == 1) {
      lingodb::scheduler::stopCurrentScheduler();
      scheduler->join();
      delete scheduler;
      scheduler = nullptr;
   }
}
