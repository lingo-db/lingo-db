#include "lingodb/utility/Tracer.h"
#include <atomic>
#include <boost/context/fiber.hpp>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <memory>
#include <thread>

#include <pthread.h>
#ifndef ASAN_ACTIVE
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Task.h"
#include "mlir/IR/MLIRContext.h"
namespace lingodb::scheduler {
class Worker;
namespace {
static thread_local Worker* currentWorker;
static constexpr size_t maxFibersPerWorker = 64;
#ifndef ASAN_ACTIVE
// Size of the PROT_NONE guard pages placed below each fiber stack and below
// the worker's main stack. Resolved from sysconf at startup because mprotect
// requires the protected range to be page-aligned, and the system page size
// differs across platforms (4 KiB on Linux x86_64, 16 KiB on macOS arm64).
static const size_t stackGuardPageSize = []() -> size_t {
   long ps = sysconf(_SC_PAGESIZE);
   return ps > 0 ? static_cast<size_t>(ps) : 4096;
}();
// Size of the worker thread's own stack, carved off the top of the mmap region.
static constexpr size_t workerMainStackSize = 2 * 1024 * 1024;
#endif
} // end namespace

struct TaskWrapper;
#ifdef ASAN_ACTIVE

class Fiber {
   //related to managing pthreads manually
   pthread_t threadHandle{};
   bool threadHandleValid = false;

   std::atomic<bool> isRunning{false};
   bool done = true;
   Worker* worker = nullptr;
   std::shared_ptr<TaskWrapper> task = nullptr;
   std::condition_variable cvMain, cvFiber;
   std::mutex mtx;

   // Function to run on this fiber thread
   std::function<void()> runFunction;

   public:
   void setup();
   void teardown();
   Fiber() = default;

   ~Fiber() {
      if (threadHandleValid) {
         // resume(); // if you need to ensure it's not parked
         pthread_join(threadHandle, nullptr);
         threadHandleValid = false;
      }
   }

   // Start (or restart) the fiber with a new task and function.
   bool run(Worker* w, std::shared_ptr<TaskWrapper> tw, std::function<void()> f) {
      worker = w;
      task = std::move(tw);
      runFunction = std::move(f);
      done = false;
      isRunning = true;

      std::unique_lock<std::mutex> lk(mtx);

      // If an older thread is still around, join it before making a new one.
      if (threadHandleValid) {
         pthread_join(threadHandle, nullptr);
         threadHandleValid = false;
      }

      // Create the pthread with a 2MB stack (>= PTHREAD_STACK_MIN).
      createThreadWithStack();

      // Wait for initial yield before we return
      cvMain.wait(lk, [&]() { return !isRunning; });

      return done;
   }

   bool resume() {
      assert(!isRunning);
      assert(!done);
      isRunning = true;
      std::unique_lock<std::mutex> lk(mtx);
      cvFiber.notify_one();
      cvMain.wait(lk, [&]() { return !isRunning.load(); });
      return done;
   }

   void yield() {
      assert(isRunning);
      assert(!done);
      std::unique_lock<std::mutex> lk(mtx);
      isRunning = false;
      teardown();
      cvMain.notify_one();
      cvFiber.wait(lk, [&]() { return isRunning.load(); });
      setup();
   }

   bool isYielded() const {
      return !isRunning.load();
   }

   Worker* getWorker() { return worker; }
   std::shared_ptr<TaskWrapper> getTask() { return task; }

   private:
   static void* threadEntry(void* arg) {
      Fiber* self = static_cast<Fiber*>(arg);

      // Set thread-local worker
      currentWorker = self->worker;

      // Call user setup
      self->setup();

      // Run the user function
      self->runFunction();

      // Mark completion and notify main
      {
         std::unique_lock<std::mutex> lk(self->mtx);
         self->isRunning = false;
         self->done = true;
         self->teardown();
         self->cvMain.notify_one();
      }
      return nullptr;
   }

   void createThreadWithStack() {
      long stackBytes = 2 << 20;

      if (stackBytes < PTHREAD_STACK_MIN)
         stackBytes = PTHREAD_STACK_MIN;

      pthread_attr_t attr;
      int rc = pthread_attr_init(&attr);
      if (rc != 0)
         throw std::runtime_error("pthread_attr_init failed");

      rc = pthread_attr_setstacksize(&attr, stackBytes);
      if (rc != 0) {
         pthread_attr_destroy(&attr);
         throw std::runtime_error("pthread_attr_setstacksize failed");
      }

      // Optional: guard page (supported on Linux & macOS)
      pthread_attr_setguardsize(&attr, 4096);

      rc = pthread_create(&threadHandle, &attr, &Fiber::threadEntry, this);
      pthread_attr_destroy(&attr);

      if (rc != 0)
         throw std::runtime_error("pthread_create failed");

      threadHandleValid = true;
   }
};

#else
class Fiber {
   public:
   static constexpr size_t stackSize = 1 << 20;

   private:
   struct LocalAllocator {
      Fiber* fiber = nullptr;

      boost::context::stack_context allocate() {
         assert(!fiber->stackAllocated);
         boost::context::stack_context sctx;
         sctx.size = stackSize;
         sctx.sp = fiber->stackBase + stackSize;
         fiber->stackAllocated = true;
         return sctx;
      }

      void deallocate(boost::context::stack_context&) noexcept {
         assert(fiber->stackAllocated);
         fiber->stackAllocated = false;
      }

      friend class Fiber;
   };

   // Points into the worker's shared mmap'd stack region — the fiber does not
   // own this memory. The worker allocates one big region up front and hands
   // out per-fiber slots of `stackSize` bytes so that pthread can advertise
   // the whole region as its stack.
   std::byte* stackBase;
   bool stackAllocated = false;
   std::atomic<bool> isRunning = false;
   bool done = true;
   boost::context::fiber fiber;
   boost::context::fiber sink;
   Worker* worker = nullptr;
   std::shared_ptr<TaskWrapper> task = nullptr;

   public:
   void setup();
   void teardown();

   bool run(Worker* worker, std::shared_ptr<TaskWrapper> taskWrapper, const std::function<void()>&& f) {
      this->worker = worker;
      this->task = taskWrapper;
      done = false;
      isRunning = true;
      fiber = boost::context::fiber{std::allocator_arg, LocalAllocator{this}, [&](boost::context::fiber&& boostSink) {
                                       sink = std::move(boostSink);
                                       setup();
                                       f();
                                       teardown();
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

   std::shared_ptr<TaskWrapper> getTask() {
      return task;
   }

   bool resume() {
      assert(!isRunning);
      assert(!done);
      isRunning = true;
      setup();
      fiber = std::move(fiber).resume();
      teardown();
      return done;
   }

   void yield() {
      assert(isRunning);
      assert(!done);
      isRunning = false;
      teardown();
      sink = std::move(sink).resume();
   }
   bool isYielded() {
      return !isRunning;
   };

   explicit Fiber(std::byte* stackBase) : stackBase(stackBase) {}

   ~Fiber() {
      assert(done);
   }
};
#endif

struct TaskWrapper {
   std::unique_ptr<Task> task;

   std::atomic<bool> coolingDown = false;
   bool finalized = false;
   std::shared_ptr<TaskWrapper> next = nullptr;
   std::shared_ptr<TaskWrapper> prev = nullptr;
   std::atomic<int64_t> yieldedFibers = 0;
   std::atomic<int64_t> nonCompletedFibers = 0;
   std::atomic<int64_t> deployedOnWorkers = 0;
   std::atomic<int64_t> returnedFromWorkers = 0;
   //this is only to be called after the task is done, returned to the scheduler from all workers, and is not anymore used in the scheduler either
   std::function<void()> onFinalize = nullptr;
   std::mutex finalizeMutex = {};

   bool done() {
      return !task->hasWork() && nonCompletedFibers.load() == 0;
   }

   bool startFiber() {
      if (task->allocateWork()) {
         nonCompletedFibers++;
         return true;
      }
      return false;
   }

   void finishFiber() {
      nonCompletedFibers--;
   }

   void yieldFiber() {
      yieldedFibers++;
   }

   void unYieldFiber() {
      yieldedFibers--;
   }
};

void Fiber::teardown() {
   assert(task && task->task);
   task->task->teardown();
}
void Fiber::setup() {
   assert(task && task->task);
   task->task->setup();
}

class Scheduler {
   size_t numWorkers;

   std::atomic<bool> shutdown{false};
#ifdef ASAN_ACTIVE
   std::vector<std::thread> workerThreads;
#else
   // Each worker thread owns one contiguous mmap'd memory region that contains
   // all its fiber stacks plus its own pthread stack. We advertise the whole
   // region to pthread via pthread_attr_setstack so that libraries like CPython
   // — which check whether the current stack pointer falls inside the pthread
   // stack range — keep working even when a fiber is active.
   struct WorkerThread {
      pthread_t handle;
      void* stackMemory = nullptr;
      size_t stackMemorySize = 0;
   };
   std::vector<WorkerThread> workerThreads;
   struct WorkerArg {
      Scheduler* scheduler;
      size_t workerId;
      std::byte* stackRegion;
   };
   static void* workerEntry(void* raw);
#endif
   // makes sure that all threads stay alive during the shutdown process
   std::mutex shutdownMutex;
   Worker* idleWorkers = nullptr;
   std::mutex taskQueueMutex;
   std::shared_ptr<TaskWrapper> taskHead = nullptr;
   std::shared_ptr<TaskWrapper> taskTail = nullptr;

   std::atomic<size_t> stoppedWorkers = 0;

   SystemContext systemContext;

   public:
   Scheduler(size_t numWorkers = std::thread::hardware_concurrency()) : numWorkers(numWorkers) {
   }
#ifndef ASAN_ACTIVE
   ~Scheduler() {
      for (auto& wt : workerThreads) {
         if (wt.stackMemory) {
            munmap(wt.stackMemory, wt.stackMemorySize);
         }
      }
   }
#endif
   size_t getNumWorkers() {
      return numWorkers;
   }

   void putWorkerToSleep(Worker* worker);

   void start();

   void stop();

   void join() {
#ifdef ASAN_ACTIVE
      for (auto& workerThread : workerThreads) {
         assert(workerThread.joinable() && "Worker thread is not joinable");
         if (workerThread.joinable()) {
            workerThread.join();
         }
      }
#else
      for (auto& wt : workerThreads) {
         pthread_join(wt.handle, nullptr);
      }
#endif
   }

   bool isShutdown() {
      return shutdown.load();
   }

   //insert task into "active task queue"
   void enqueueTask(std::shared_ptr<TaskWrapper> wrapper);

   void enqueueTask(std::unique_ptr<Task>&& task) {
      auto wrapper = std::make_shared<TaskWrapper>();
      wrapper->task = std::move(task);
      wrapper->onFinalize = []() {};
      enqueueTask(wrapper);
   }

   //remove task from "active task queue"
   void dequeueTaskLocked(std::shared_ptr<TaskWrapper> task) {
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
      task->prev = nullptr;
      task->next = nullptr;
   }

   std::shared_ptr<TaskWrapper> getTask() {
      std::lock_guard<std::mutex> lock(taskQueueMutex);
      auto potentialTask = taskHead;
      size_t minYieldCount = std::numeric_limits<size_t>::max();
      std::shared_ptr<TaskWrapper> minYieldTask = nullptr;
      while (potentialTask) {
         //if we see a task that has no work left, we move it to the cooling down queue
         if (!potentialTask->task->hasWork()) {
            auto toCoolDown = potentialTask;
            potentialTask = potentialTask->next;
            dequeueTaskLocked(toCoolDown);
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
   void returnTask(std::shared_ptr<TaskWrapper> task) {
      // Multiple worker may call here concurrently. Avoid one worker already called `delete task`
      // but another worker is still at `task->finalized`
      auto returnedNum = task->returnedFromWorkers.fetch_add(1) + 1;
      if (!task->task->hasWork()) {
         {
            std::lock_guard<std::mutex> lock(taskQueueMutex);
            if (!task->coolingDown) {
               task->coolingDown = true;
               dequeueTaskLocked(task);
            }
         }
         // Since the task is cooling down (protected with taskQueueMutex), deployedOnWorkers can not be changed anymore
         auto taskDeployed = task->deployedOnWorkers.load();
         if (taskDeployed == returnedNum) {
            std::unique_lock<std::mutex> lk(task->finalizeMutex);
            assert(!task->finalized);
            task->finalized = true;
            task->onFinalize();
         }
      }
   }
   SystemContext& getSystemContext() {
      return systemContext;
   }
};

class Worker {
   class FiberAllocator {
      static constexpr size_t initiallyAllocated = 2;
      size_t numAllocated;
      size_t maxFibers;
#ifndef ASAN_ACTIVE
      // Base of the worker's mmap'd fiber-stack region. Slot k lives at
      // stackRegion + k * Fiber::stackSize. The memory is owned by Scheduler,
      // not by the allocator.
      std::byte* stackRegion;
#endif
      std::deque<std::unique_ptr<Fiber>> allocatedAndAvailableFibers;

      public:
#ifdef ASAN_ACTIVE
      explicit FiberAllocator(size_t maxFibers) : numAllocated(initiallyAllocated), maxFibers(maxFibers) {
         for (size_t i = 0; i < initiallyAllocated; i++) {
            allocatedAndAvailableFibers.push_back(std::make_unique<Fiber>());
         }
      }
#else
      // Slot layout within `stackRegion` (low → high):
      //   [ guard | fiber 0 | guard | fiber 1 | ... | guard | fiber N-1 ]
      // The stride between fiber k's usable stack and fiber k+1's usable stack
      // is stackSize + stackGuardPageSize, and fiber k's usable stack starts
      // stackGuardPageSize bytes into its slot (past the guard page below it).
      static std::byte* slotStackBase(std::byte* region, size_t k) {
         return region + k * (Fiber::stackSize + stackGuardPageSize) + stackGuardPageSize;
      }

      FiberAllocator(std::byte* stackRegion, size_t maxFibers) : numAllocated(initiallyAllocated), maxFibers(maxFibers), stackRegion(stackRegion) {
         for (size_t i = 0; i < initiallyAllocated; i++) {
            allocatedAndAvailableFibers.push_back(std::make_unique<Fiber>(slotStackBase(stackRegion, i)));
         }
      }
#endif

      bool canAllocate() {
         return numAllocated < maxFibers || !allocatedAndAvailableFibers.empty();
      }

      std::unique_ptr<Fiber> allocate() {
         if (allocatedAndAvailableFibers.empty()) {
            if (numAllocated < maxFibers) {
#ifdef ASAN_ACTIVE
               numAllocated++;
               return std::make_unique<Fiber>();
#else
               auto* slot = slotStackBase(stackRegion, numAllocated);
               numAllocated++;
               return std::make_unique<Fiber>(slot);
#endif
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

   std::unordered_map<std::shared_ptr<TaskWrapper>, std::unique_ptr<Fiber>> waitingOnTasks;
   std::atomic<size_t> numWaitingFibers = 0;

   std::shared_ptr<TaskWrapper> currentTask = nullptr;

   using TimePoint = std::chrono::time_point<std::chrono::system_clock>;
   TimePoint startWaitTime = TimePoint::min();

   public:
   //for cheaply collecting idle workers
   Worker* nextIdleWorker = nullptr;
   bool isInIdleList = false;
   // for sleeping / waking up if there is nothing to do
   std::mutex mutex;
   std::condition_variable cv;
   size_t workerId;
   bool allowedToSleep = true;
   bool shouldSleep = true;

#ifdef ASAN_ACTIVE
   Worker(Scheduler& scheduler, size_t id) : scheduler(scheduler), fiberAllocator(maxFibersPerWorker), workerId(id) {
   }
#else
   Worker(Scheduler& scheduler, size_t id, std::byte* stackRegion) : scheduler(scheduler), fiberAllocator(stackRegion, maxFibersPerWorker), workerId(id) {
   }
#endif

   void wakeupWorker() {
      {
         std::unique_lock<std::mutex> lock(mutex);
         allowedToSleep = false;
         shouldSleep = false;
         cv.notify_one();
      }
   }

   void yieldIfRunning() {
      if (currentFiber) {
         currentFiber->yield();
      }
   }

   void awaitChildTask(std::unique_ptr<Task> task) {
      auto taskWrapper = std::make_shared<TaskWrapper>();
      taskWrapper->task = std::move(task);
      Fiber* toYield;
      {
         std::unique_lock<std::mutex> fiberLock(fiberMutex);
         waitingOnTasks[taskWrapper] = std::move(currentFiber);
         numWaitingFibers++;
         toYield = waitingOnTasks[taskWrapper].get();
         taskWrapper->onFinalize = [&] {
            {
               std::unique_lock<std::mutex> fiberLock2(fiberMutex);
               assert(waitingOnTasks.contains(taskWrapper));
               assert(waitingOnTasks[taskWrapper]);
               while (!waitingOnTasks[taskWrapper]->isYielded()) {}
               runnableFibers.push_back(std::move(waitingOnTasks[taskWrapper]));
               waitingOnTasks.erase(taskWrapper);
               numWaitingFibers--;
               // taskWrapper will be automatically destroyed by shared_ptr
            }
            wakeupWorker();
         };
         scheduler.enqueueTask(taskWrapper);
      }
      toYield->yield();
      assert(taskWrapper->finalized);
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
            auto task = currentFiber->getTask();
            task->finishFiber();
            fiberAllocator.deallocate(std::move(currentFiber));
         };
         if (currentFiber) {
            if (currentFiber->resume()) {
               //unyield because it was previously registered as yielded
               if (currentFiber->getTask()) {
                  currentFiber->getTask()->unYieldFiber();
               }
               auto resumeTask = currentFiber->getTask();
               handleFiberComplete();
               scheduler.returnTask(resumeTask);
            }
            // we just finished a fiber, so see if we have more runnable fibers
            continue;
         }
         if (fiberAllocator.canAllocate()) {
            std::shared_ptr<TaskWrapper> currTask = nullptr;
            {
               std::unique_lock<std::mutex> lock(mutex);
               if (this->currentTask) {
                  currTask = this->currentTask;
                  this->currentTask = nullptr;
               }
            }
            if (currTask && !currTask->task->hasWork()) {
               scheduler.returnTask(currTask); //this usage is ok
               currTask = nullptr;
               continue;
            }
            if (!currTask) {
               currTask = scheduler.getTask();
            }

            if (currTask) {
               // Step 1. try startFiber, this will try to allocate work for the task
               if (!currTask->startFiber()) {
                  scheduler.returnTask(currTask); // this usage is okay
                  continue;
               }
               //work on (part of) (new) task
               currentFiber = fiberAllocator.allocate();
               assert(currentFiber);
               // Step 3. consume reserved work
               auto fiberDone = currentFiber->run(this, currTask, [&] {
                  currTask->task->performWork();
               });
               if (fiberDone) {
                  this->startWaitTime = TimePoint::min();
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
                  scheduler.returnTask(currTask); //this usage is ok
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
static std::atomic<size_t> numSchedulerUsers = 0;

void stopCurrentScheduler() {
   if (scheduler) {
      scheduler->stop();
   }
}
} // end namespace

#ifndef ASAN_ACTIVE
void* Scheduler::workerEntry(void* raw) {
   std::unique_ptr<WorkerArg> arg(static_cast<WorkerArg*>(raw));
#if defined(__APPLE__) && defined(__arm64__)
   // on apple silicon: signal os to prefer performance cores for scheduling this thread
   // https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html
   // we silently ignore errors, since this setting is not mission-critical.
   pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
#ifdef TRACER
   utility::Tracer::ensureThreadLocalTraceRecordList();
#endif
   Worker worker(*arg->scheduler, arg->workerId, arg->stackRegion);
   currentWorker = &worker;
   worker.work();
   arg->scheduler->stoppedWorkers++;
   currentWorker = nullptr;
   std::unique_lock<std::mutex> lock(arg->scheduler->shutdownMutex);
   return nullptr;
}
#endif

void Scheduler::start() {
   scheduler = this;
   for (size_t i = 0; i < numWorkers; i++) {
#ifdef ASAN_ACTIVE
      workerThreads.emplace_back([this, i] {
#if defined(__APPLE__) && defined(__arm64__)
         // on apple silicon: signal os to prefer performance cores for scheduling this thread
         // https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html
         // we silently ignore errors, since this setting is not mission-critical.
         pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
#ifdef TRACER
         utility::Tracer::ensureThreadLocalTraceRecordList();
#endif
         Worker worker(*this, i);
         currentWorker = &worker;
         worker.work();
         stoppedWorkers++;
         currentWorker = nullptr;
         std::unique_lock<std::mutex> lock(shutdownMutex);
      });
#else
      // Layout (low → high):
      //   [ G | fiber 0 | G | fiber 1 | ... | G | fiber N-1 | G | worker main ]
      // where each `G` is a PROT_NONE guard page. One guard below every fiber
      // slot traps that fiber's stack overflow before it can corrupt the
      // adjacent fiber; the trailing guard protects the worker's main stack
      // from overflowing downwards into the last fiber. pthread starts SP at
      // the very top of the region, and the whole range is declared as the
      // thread's stack via pthread_attr_setstack, so CPython's stack-range
      // check sees the SP as in-range regardless of which fiber is active.
      const size_t fiberSlotStride = Fiber::stackSize + stackGuardPageSize;
      const size_t fiberRegionSize = maxFibersPerWorker * fiberSlotStride;
      const size_t totalSize = fiberRegionSize + stackGuardPageSize + workerMainStackSize;

      void* region = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (region == MAP_FAILED) {
         throw std::runtime_error("mmap failed for worker stack region");
      }
      auto protectGuard = [region, totalSize](size_t offset) {
         if (mprotect(static_cast<char*>(region) + offset, stackGuardPageSize, PROT_NONE) != 0) {
            munmap(region, totalSize);
            throw std::runtime_error("mprotect failed for stack guard page");
         }
      };
      // Guard page below every fiber's usable stack.
      for (size_t k = 0; k < maxFibersPerWorker; ++k) {
         protectGuard(k * fiberSlotStride);
      }
      // Guard page below the worker's main stack.
      protectGuard(fiberRegionSize);

      pthread_attr_t attr;
      if (pthread_attr_init(&attr) != 0) {
         munmap(region, totalSize);
         throw std::runtime_error("pthread_attr_init failed");
      }
      if (pthread_attr_setstack(&attr, region, totalSize) != 0) {
         pthread_attr_destroy(&attr);
         munmap(region, totalSize);
         throw std::runtime_error("pthread_attr_setstack failed");
      }

      auto* arg = new WorkerArg{this, i, static_cast<std::byte*>(region)};
      pthread_t handle;
      int rc = pthread_create(&handle, &attr, &Scheduler::workerEntry, arg);
      pthread_attr_destroy(&attr);
      if (rc != 0) {
         delete arg;
         munmap(region, totalSize);
         throw std::runtime_error("pthread_create failed");
      }
      workerThreads.push_back({handle, region, totalSize});
#endif
   }
}

void Scheduler::stop() {
   std::unique_lock<std::mutex> lock(shutdownMutex);
   shutdown.store(true);
   size_t cntr = 0;
   while (stoppedWorkers.load() < numWorkers) {
      usleep(100);
      {
         std::unique_lock<std::mutex> lock(taskQueueMutex);
         while (idleWorkers) {
            assert(cntr++ < numWorkers);
            auto* currWorker = idleWorkers;

            std::unique_lock<std::mutex> workerLock(currWorker->mutex);
            currWorker->isInIdleList = false;
            idleWorkers = currWorker->nextIdleWorker;
            currWorker->shouldSleep = false;
            currWorker->cv.notify_one();
         }
      }
   }
}

void Scheduler::enqueueTask(std::shared_ptr<TaskWrapper> wrapper) {
   std::lock_guard<std::mutex> lock(taskQueueMutex);
   if (taskTail) {
      taskTail->next = wrapper;
      wrapper->prev = taskTail;
      taskTail = wrapper;
   } else {
      taskHead = wrapper;
      taskTail = wrapper;
   }

   size_t cntr = 0;
   while (idleWorkers) {
      assert(cntr++ < numWorkers);
      Worker* currWorker = idleWorkers;
      std::unique_lock<std::mutex> workerLock(currWorker->mutex);
      currWorker->isInIdleList = false;
      idleWorkers = idleWorkers->nextIdleWorker;
      currWorker->shouldSleep = false;
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
      //assertRelease(!worker->isInIdleList, "Worker already in idle list");
      if (!worker->isInIdleList) {
         worker->nextIdleWorker = idleWorkers;
         idleWorkers = worker;
         worker->isInIdleList = true;
      }
      lock.unlock();
      worker->shouldSleep = true;
      worker->cv.wait(workerLock, [&]() { return !worker->shouldSleep; });
   } else {
      worker->allowedToSleep = true;
   }
   //assertRelease(!worker->isInIdleList, "Worker still in idle list after wakeup");
}

void awaitEntryTask(std::unique_ptr<Task> task) {
   auto taskWrapper = std::make_shared<TaskWrapper>();
   taskWrapper->task = std::move(task);
   std::condition_variable cvFinished;
   bool finished = false;
   taskWrapper->onFinalize = [&]() {
      finished = true;
      cvFinished.notify_one();
   };
   std::unique_lock<std::mutex> lk(taskWrapper->finalizeMutex);
   scheduler->enqueueTask(taskWrapper);
   cvFinished.wait(lk, [&]() { return finished; });
   // taskWrapper will be automatically destroyed when it goes out of scope
   lk.release();
}
void awaitChildTask(std::unique_ptr<Task> task) {
   currentWorker->awaitChildTask(std::move(task));
}
void enqueueTask(std::unique_ptr<Task> task) {
   scheduler->enqueueTask(std::move(task));
}
std::unique_ptr<SchedulerHandle> startScheduler(size_t numWorkers) {
#if defined(__APPLE__) && defined(__arm64__)
   // on apple silicon: set QOS_CLASS_USER_INTERACTIVE for main thread on scheduler startup
   // we silently ignore errors here, since this setting is not mission-critical.
   pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

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
SystemContext& getSystemContext() {
   return scheduler->getSystemContext();
}
void yieldCurrentTask() {
   if (currentWorker) {
      currentWorker->yieldIfRunning();
   }
}
SystemContext::~SystemContext() {
#ifndef MLIR_DISABLED
   std::lock_guard<std::mutex> lock(contextStackMutex);
   while (!llvmContextStack.empty()) {
      auto* ctx = llvmContextStack.top();
      llvmContextStack.pop();
      delete ctx;
   }
   while (!noLlvmContextStack.empty()) {
      auto* ctx = noLlvmContextStack.top();
      noLlvmContextStack.pop();
      delete ctx;
   }
#endif
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
