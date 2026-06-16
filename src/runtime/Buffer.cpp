#include "lingodb/runtime/Buffer.h"
#include "lingodb/scheduler/Tasks.h"
#include "lingodb/utility/Tracer.h"
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <shared_mutex>
namespace {
static lingodb::utility::Tracer::Event chunkedBufferScan("ChunkedBuffer", "bufferScan");
static lingodb::utility::Tracer::Event bufferScan("Buffer", "bufferScan");
static lingodb::utility::Tracer::Event chunkedBufferChunk("BufferIterator", "chunk", false);
static lingodb::utility::Tracer::Event bufferChunk("BufferIterator", "chunk", false);

class FlexibleBufferWorkerResvState {
   public:
   size_t bufferId;
   std::shared_mutex mutex;
   size_t resvCursor{0};
   size_t resvId{0};
   size_t unitAmount;
   // workerId steal task from
   size_t stealWorkerId{std::numeric_limits<size_t>::max()};

   bool hasMoreWork() {
      std::shared_lock<std::shared_mutex> resvLock(mutex);
      return resvCursor < unitAmount;
   }

   int fetchAndNextOwn(size_t splitSize, const std::vector<lingodb::runtime::Buffer>& buffers, std::atomic<size_t>& startIndex) {
      std::unique_lock<std::shared_mutex> resvLock(mutex);
      size_t cur = resvCursor++;
      if (cur < unitAmount) {
         return cur;
      }
      //no work left in current buffer, try to fetch next buffer
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex < buffers.size()) {
         auto& buffer = buffers[localStartIndex];
         unitAmount = (buffer.numElements + splitSize - 1) / splitSize;
         resvCursor = 1;
         resvId = 0;
         bufferId = localStartIndex;
         return 0;
      }
      return -1;
   }

   // Reserve a unit when stealing. The victim's bufferId is captured under the
   // same lock so the stealer records a consistent (bufferId, unitId) pair;
   // reading it later would race with the victim advancing to its next buffer.
   int fetchAndNext(size_t& outBufferId) {
      std::unique_lock<std::shared_mutex> resvLock(mutex);
      size_t cur = resvCursor++;
      if (cur >= unitAmount) return -1;
      outBufferId = bufferId;
      return cur;
   }
};

class FlexibleBufferIteratorTask : public lingodb::scheduler::TaskWithImplicitContext {
   std::vector<lingodb::runtime::Buffer>& buffers;
   size_t typeSize;
   const std::function<void(lingodb::runtime::Buffer)> cb;
   std::atomic<size_t> startIndex{0};
   size_t splitSize{200};
   std::vector<std::unique_ptr<FlexibleBufferWorkerResvState>> workerResvs;

   public:
   FlexibleBufferIteratorTask(std::vector<lingodb::runtime::Buffer>& buffers, size_t typeSize, const std::function<void(lingodb::runtime::Buffer)> cb) : buffers(buffers), typeSize(typeSize), cb(cb) {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         workerResvs.emplace_back(std::make_unique<FlexibleBufferWorkerResvState>());
      }
   }
   void unitRun(size_t bufferId, int unitId) {
      auto& buffer = buffers[bufferId];
      if (unitId < 0) {
         return;
      }
      lingodb::utility::Tracer::Trace trace(chunkedBufferChunk);
      size_t begin = splitSize * unitId;
      size_t len = std::min(begin + splitSize, static_cast<size_t>(buffer.numElements)) - begin;
      auto buf = lingodb::runtime::Buffer{len, buffer.ptr + begin * std::max(1ul, typeSize)};
      cb(buf);
      trace.stop();
   }

   bool allocateWork() override {
      // quick check for exhaust. workExhausted is true if there is no more buffer or no more
      // work unit in own local state or steal from other workers.
      if (workExhausted.load()) {
         return false;
      }

      //1. if the current worker has more work locally, do it
      auto* state = workerResvs[lingodb::scheduler::currentWorkerId()].get();
      auto id = state->fetchAndNextOwn(splitSize, buffers, startIndex);
      if (id != -1) {
         state->resvId = id;
         return true;
      }
      //3. if the current worker has no more work locally and no more work globally, try to steal work from the worker we stole from last time
      if (state->stealWorkerId != std::numeric_limits<size_t>::max()) {
         auto* other = workerResvs[state->stealWorkerId].get();
         if (other->hasMoreWork()) {
            size_t stolenBufferId;
            auto id = other->fetchAndNext(stolenBufferId);
            if (id != -1) {
               state->resvId = id;
               state->bufferId = stolenBufferId;
               return true;
            }
         }
         state->stealWorkerId = std::numeric_limits<size_t>::max();
      }
      //4. if the current worker has no more work locally and no more work globally, try to steal work from other workers
      for (size_t i = 1; i < workerResvs.size(); i++) {
         // make sure index of worker to steal never exceed worker number limits
         auto idx = (lingodb::scheduler::currentWorkerId() + i) % workerResvs.size();
         auto* other = workerResvs[idx].get();
         if (other->hasMoreWork()) {
            size_t stolenBufferId;
            auto id = other->fetchAndNext(stolenBufferId);
            if (id != -1) {
               // only current worker can modify its onw stealWorkerId. no need to lock
               state->stealWorkerId = idx;
               state->resvId = id;
               state->bufferId = stolenBufferId;
               return true;
            }
         }
      }

      workExhausted.store(true);
      return false;
   }
   void performWork() override {
      auto* state = workerResvs[lingodb::scheduler::currentWorkerId()].get();
      unitRun(state->bufferId, state->resvId);
   }
};

class BufferIteratorTask : public lingodb::scheduler::TaskWithImplicitContext {
   lingodb::runtime::Buffer& buffer;
   size_t bufferLen;
   void* contextPtr;
   const std::function<void(lingodb::runtime::Buffer, size_t, size_t, void*)> cb;
   size_t splitSize{20000};
   std::atomic<size_t> startIndex{0};
   std::vector<size_t> workerResvs;

   public:
   BufferIteratorTask(lingodb::runtime::Buffer& buffer, size_t typeSize, void* contextPtr, const std::function<void(lingodb::runtime::Buffer, size_t, size_t, void*)> cb) : buffer(buffer), bufferLen(buffer.numElements / typeSize), contextPtr(contextPtr), cb(cb) {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         workerResvs.push_back(0);
      }
   }
   bool allocateWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex * splitSize >= bufferLen) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }
   void performWork() override {
      auto localStartIndex = workerResvs[lingodb::scheduler::currentWorkerId()];
      auto begin = localStartIndex * splitSize;
      auto end = (localStartIndex + 1) * splitSize;
      if (end > bufferLen) {
         end = bufferLen;
      }
      lingodb::utility::Tracer::Trace trace(bufferChunk);
      cb(buffer, begin, end, contextPtr);
      trace.stop();
   }
};

} // end namespace

bool lingodb::runtime::BufferIterator::isIteratorValid(lingodb::runtime::BufferIterator* iterator) {
   return iterator->isValid();
}
void lingodb::runtime::BufferIterator::iteratorNext(lingodb::runtime::BufferIterator* iterator) {
   iterator->next();
}
lingodb::runtime::Buffer lingodb::runtime::BufferIterator::iteratorGetCurrentBuffer(lingodb::runtime::BufferIterator* iterator) {
   return iterator->getCurrentBuffer();
}
void lingodb::runtime::BufferIterator::destroy(lingodb::runtime::BufferIterator* iterator) {
   delete iterator;
}
// Below this many elements a parallel scan is run inline on the calling worker
// instead of spawning a task. Dispatching a morsel task wakes every worker, has
// them contend for a handful of 200-element morsels, then sleep again; for the
// tiny per-iteration working sets of graph algorithms that wakeup/sync overhead
// dwarfs the actual work. Running inline also means only one thread-local state
// is populated, so the subsequent merge step is free. Tunable via env var.
size_t lingodb::runtime::getParallelScanThreshold() {
   static size_t threshold = [] {
      if (const char* e = std::getenv("LINGODB_PARALLEL_THRESHOLD")) {
         return static_cast<size_t>(std::strtoull(e, nullptr, 10));
      }
      return static_cast<size_t>(400);
   }();
   return threshold;
}

void lingodb::runtime::FlexibleBuffer::iterateBuffersParallel(const std::function<void(Buffer)>& fn) {
   if (totalLen < getParallelScanThreshold()) {
      for (auto& buffer : buffers) {
         fn(buffer);
      }
      return;
   }
   lingodb::scheduler::awaitChildTask(std::make_unique<FlexibleBufferIteratorTask>(buffers, typeSize, fn));
}
class FlexibleBufferIterator : public lingodb::runtime::BufferIterator {
   lingodb::runtime::FlexibleBuffer& flexibleBuffer;
   size_t currBuffer;

   public:
   FlexibleBufferIterator(lingodb::runtime::FlexibleBuffer& flexibleBuffer) : flexibleBuffer(flexibleBuffer), currBuffer(0) {}
   bool isValid() override {
      return currBuffer < flexibleBuffer.getBuffers().size();
   }
   void next() override {
      currBuffer++;
   }
   lingodb::runtime::Buffer getCurrentBuffer() override {
      lingodb::runtime::Buffer orig = flexibleBuffer.getBuffers().at(currBuffer);
      return lingodb::runtime::Buffer{orig.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), orig.ptr};
   }
   void iterateEfficient(bool parallel, void (*forEachChunk)(lingodb::runtime::Buffer, void*), void* contextPtr) override {
      if (parallel) {
         flexibleBuffer.iterateBuffersParallel([&](lingodb::runtime::Buffer buffer) {
            buffer = lingodb::runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            lingodb::utility::Tracer::Trace trace(chunkedBufferChunk);
            forEachChunk(buffer, contextPtr);
         });
      } else {
         for (auto buffer : flexibleBuffer.getBuffers()) {
            buffer = lingodb::runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            lingodb::utility::Tracer::Trace trace(chunkedBufferChunk);
            forEachChunk(buffer, contextPtr);
         }
      }
   }
};

lingodb::runtime::BufferIterator* lingodb::runtime::FlexibleBuffer::createIterator() {
   auto* it = new FlexibleBufferIterator(*this);
   getCurrentExecutionContext()->registerState({it, [](void* ptr) { delete reinterpret_cast<BufferIterator*>(ptr); }});
   return it;
}
size_t lingodb::runtime::FlexibleBuffer::getLen() const {
   return totalLen;
}

void lingodb::runtime::BufferIterator::iterate(lingodb::runtime::BufferIterator* iterator, bool parallel, void (*forEachChunk)(lingodb::runtime::Buffer, void*), void* contextPtr) {
   utility::Tracer::Trace trace(chunkedBufferScan);
   iterator->iterateEfficient(parallel, forEachChunk, contextPtr);
}

void lingodb::runtime::Buffer::iterate(bool parallel, lingodb::runtime::Buffer buffer, size_t typeSize, void (*forEachChunk)(lingodb::runtime::Buffer, size_t, size_t, void*), void* contextPtr) {
   utility::Tracer::Trace trace(bufferScan);
   if (parallel && buffer.numElements / typeSize >= getParallelScanThreshold()) {
      lingodb::scheduler::awaitChildTask(std::make_unique<BufferIteratorTask>(buffer, typeSize, contextPtr, forEachChunk));
   } else {
      lingodb::utility::Tracer::Trace trace2(bufferChunk);
      forEachChunk(buffer, 0, buffer.numElements / typeSize, contextPtr);
   }
}
