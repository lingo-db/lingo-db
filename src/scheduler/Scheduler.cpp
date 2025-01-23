#include "lingodb/utility/Tracer.h"
#include <atomic>
#include <boost/context/fiber.hpp>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <memory>
#include <syncstream>
#include <thread>

#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Task.h"

namespace lingodb::scheduler {
class Worker;

class TaskWrapper;

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
      //todo: maybe cleanup
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
   std::unique_ptr<Fiber> waitingOnTaskCompletion;

   void finalize();

   bool done() {
      return !task->hasWork() && nonCompletedFibers.load() == 0;
   }

   void startFiber() {
      nonCompletedFibers++;
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

class SchedulerImpl : public Scheduler {
   std::atomic<bool> shutdown{false};
   std::vector<std::thread> workerThreads;
   Worker* idleWorkers = nullptr;
   std::mutex taskQueueMutex;
   TaskWrapper* taskHead = nullptr;
   TaskWrapper* taskTail = nullptr;

   //queue for tasks that are cooling down i.e. they don't have work left to do, but are also not yet finished.
   std::mutex coolingDownMutex;
   TaskWrapper* coolingDownHead = nullptr;
   TaskWrapper* coolingDownTail = nullptr;

   public:
   SchedulerImpl(size_t numWorkers = std::thread::hardware_concurrency()) : Scheduler(numWorkers) {
   }

   void putWorkerToSleep(Worker* worker);

   virtual void start() override;

   void stop();

   virtual void join() override {
      for (auto& workerThread : workerThreads) {
         workerThread.join();
      }
   }

   bool isShutdown() {
      return shutdown.load();
   }

   //insert task into "active task queue"
   void enqueueTask(TaskWrapper* wrapper);

   virtual void enqueueTask(std::unique_ptr<Task>&& task) override {
      enqueueTask(new TaskWrapper{std::move(task)});
   }

   //remove task from "active task queue"
   void dequeueTaskLocked(TaskWrapper* task) {
      if (task->prev) {
         task->prev->next = task->next;
      } else {
         taskHead = task->next;
      }
      if (task->next) {
         task->next->prev = task->prev;
      } else {
         taskTail = task->prev;
      }
   }

   TaskWrapper* getTask() {
      std::lock_guard lock(taskQueueMutex);
      auto potentialTask = taskHead;
      size_t minYieldCount = std::numeric_limits<size_t>::max();
      TaskWrapper* minYieldTask = nullptr;
      while (potentialTask) {
         //if we see a task that has no work left, we move it to the cooling down queue
         if (!potentialTask->task->hasWork()) {
            auto toCoolDown = potentialTask;
            potentialTask = potentialTask->next;
            dequeueTaskLocked(toCoolDown);
            std::lock_guard lock1(coolingDownMutex);
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
            if (yieldedFibers < minYieldCount) {
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
      task->deployedOnWorkers--;
      if (task->finalized) {
         if (task->deployedOnWorkers.load() == 0) {
            delete task;
         }
      }
   }

   void finalizeTask(TaskWrapper* task) {
      if (task->coolingDown) {
         //simple case: already in cooling down queue
         // -> only need to lock cooling down queue
         std::lock_guard lock(coolingDownMutex);
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
            std::lock_guard lock(taskQueueMutex);
            alreadyCoolingDown = task->coolingDown;
            if (!alreadyCoolingDown) {
               dequeueTaskLocked(task);
            }
         }
         if (alreadyCoolingDown) {
            std::lock_guard lock(coolingDownMutex);
            if (coolingDownTail) {
               coolingDownTail->next = task;
               task->prev = coolingDownTail;
               coolingDownTail = task;
            } else {
               coolingDownHead = task;
               coolingDownTail = task;
            }
         }
      }
      task->finalize();
      task->finalized = true;
   }
};

SchedulerImpl* scheduler;

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
            return std::move(fiber);
         }
      }

      void deallocate(std::unique_ptr<Fiber>&& fiber) {
         allocatedAndAvailableFibers.push_back(std::move(fiber));
      }
   };

   SchedulerImpl& scheduler;

   //for managing a pool of fibers/stacks
   FiberAllocator fiberAllocator;

   //all fibers that are currently runnable
   std::mutex fiberMutex;
   std::deque<std::unique_ptr<Fiber>> runnableFibers;

   std::unique_ptr<Fiber> currentFiber;

   //todo: local task
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

   Worker(SchedulerImpl& scheduler, size_t id) : scheduler(scheduler), fiberAllocator(64), workerId(id) {
   }

   void wakeupFiber(std::unique_ptr<Fiber>&& fiber) {
      {
         std::lock_guard fiberQueueLock(fiberMutex);
         runnableFibers.push_back(std::move(fiber));
      }
      {
         std::unique_lock lock(mutex);
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
            std::lock_guard lock(fiberMutex);

            if (!runnableFibers.empty()) {
               currentFiber = std::move(runnableFibers.front());
               runnableFibers.pop_front();
            }
         }
         auto handleFiberComplete = [&]() {
            auto task = currentFiber->getTask();
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
               handleFiberComplete();
            }
            assert(!currentFiber);
         }
         if (fiberAllocator.canAllocate()) {
            TaskWrapper* currTask = nullptr;
            {
               std::unique_lock lock(mutex);
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
               //work on (part of) (new) task
               currentFiber = fiberAllocator.allocate();
               assert(currentFiber);
               currTask->startFiber();
               auto fiberDone = currentFiber->run(this, currTask, [&] {
                  currTask->task->run();
               });
               if (fiberDone) {
                  handleFiberComplete();
               } else {
                  currTask->yieldFiber();
               }
               if (currTask->task->hasWork()) {
                  std::lock_guard lock(mutex);
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

static thread_local Worker* currentWorker;

void SchedulerImpl::start() {
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

void SchedulerImpl::stop() {
   shutdown.store(true);
   std::unique_lock lock(taskQueueMutex);
   size_t cntr = 0;
   while (idleWorkers) {
      assert(cntr++ < numWorkers);
      auto currWorker = idleWorkers;
      currWorker->isInIdleList = false;
      idleWorkers = currWorker->nextIdleWorker;
      currWorker->cv.notify_one();
   }
}

void SchedulerImpl::enqueueTask(TaskWrapper* wrapper) {
   {
      std::lock_guard lock(taskQueueMutex);
      if (taskTail) {
         taskTail->next = wrapper;
         wrapper->prev = taskTail;
         taskTail = wrapper;
      } else {
         taskHead = wrapper;
         taskTail = wrapper;
      }
   }
   std::lock_guard lock(taskQueueMutex);
   size_t cntr = 0;
   while (idleWorkers) {
      assert(cntr++ < numWorkers);
      Worker* currWorker = idleWorkers;
      currWorker->isInIdleList = false;
      idleWorkers = idleWorkers->nextIdleWorker;

      currWorker->cv.notify_one();
   }
}

void SchedulerImpl::putWorkerToSleep(Worker* worker) {
   if (isShutdown()) {
      return;
   }
   std::unique_lock lock(taskQueueMutex);
   std::unique_lock workerLock(worker->mutex);
   if (worker->allowedToSleep) {
      //std::cout << "Putting worker to sleep " << worker->workerId << std::endl;
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
   //std::cout << "Waking up worker " << worker->workerId << std::endl;
}

void TaskWrapper::finalize() {
   if (waitingOnTaskCompletion) {
      //std::cout << "Finalizing Task on Worker " << currentWorker->workerId << std::endl;
      auto worker = waitingOnTaskCompletion->getWorker();
      assert(worker);
      worker->wakeupFiber(std::move(waitingOnTaskCompletion));
   }
}

void awaitChildTask(std::unique_ptr<Task> task) {
   currentWorker->awaitChildTask(std::move(task));
}
std::unique_ptr<Scheduler> createScheduler(size_t numWorkers) {
   return std::make_unique<SchedulerImpl>(numWorkers);
}
void stopCurrentScheduler() {
   if (scheduler) {
      scheduler->stop();
   }
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

/*
int main() {
   std::signal(SIGINT, signalHandler);
   std::signal(SIGTERM, signalHandler);
   Scheduler scheduler(8);
   scheduler.start();
   scheduler.enqueueTask(std::make_unique<QueryExecutionTask>());
   scheduler.join();
   utility::Tracer::dump();

   return 0;
}*/
