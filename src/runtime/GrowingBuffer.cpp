#include "lingodb/runtime/GrowingBuffer.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/utility/Tracer.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <queue>
namespace {
static utility::Tracer::Event createEvent("GrowingBuffer", "create");
static utility::Tracer::Event mergeEvent("GrowingBuffer", "merge");
static utility::Tracer::Event sortEvent("GrowingBuffer", "sort");
// static utility::Tracer::Event rawSortEvent("GrowingBuffer", "rawSort");
static utility::Tracer::Event sortCopyEvent("GrowingBuffer", "sortCopy");
static utility::Tracer::Event sortLocalEvent("GrowingBuffer", "sortLocal");
static utility::Tracer::Event sortSepSyncEvent("GrowingBuffer", "sortSepSync");
static utility::Tracer::Event sortSepSearchEvent("GrowingBuffer", "sortSepSearch");
static utility::Tracer::Event sortOutputRangeEvent("GrowingBuffer", "sortOutputRange");
static utility::Tracer::Event sortMergeEvent("GrowingBuffer", "sortMerge");

class SortLocalTaskState {
public:
   size_t begin;
   size_t end;
   // TODO RENAME TO seperators and remove globalSeperators
   std::vector<size_t> localSeperators;
   std::vector<size_t> globalSeperators;
};


class SortLocalTask : public lingodb::scheduler::Task {
   // TODO the following 6 members could form a struct like sortContext
   std::vector<uint8_t*>& input;
   size_t seperatorCnt;
   size_t splitCnt;
   size_t splitSize;
   bool (*compareFn)(uint8_t*, uint8_t*);
   std::vector<SortLocalTaskState>& localStates;
   std::atomic<size_t> startIndex{0};
   std::vector<size_t> workerResvs;

   public:
   SortLocalTask(std::vector<uint8_t*>& input, size_t seperatorCnt, size_t splitCnt, bool (*compareFn)(uint8_t*, uint8_t*), std::vector<SortLocalTaskState> &localStates) : input(input), seperatorCnt(seperatorCnt), splitCnt(splitCnt), compareFn(compareFn), localStates(localStates) {
      splitSize = (input.size()+splitCnt-1)/splitCnt;
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         workerResvs.push_back(0);
      }
      for (size_t i = 0; i < splitCnt; i ++) {
         std::vector<size_t> sepVec;
         localStates.push_back(SortLocalTaskState(0,0,sepVec));
      }
   }
   bool reserveWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex >= splitCnt) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }
   void consumeWork() override {
      utility::Tracer::Trace trace1(sortLocalEvent);
      auto localStartIndex = workerResvs[lingodb::scheduler::currentWorkerId()];
      auto begin = localStartIndex * splitSize;
      auto end = (localStartIndex + 1) * splitSize;
      if (end > input.size()) {
         end = input.size();
      }
      std::sort(input.begin() + begin, input.begin() + end, compareFn);

      auto range = end - begin;
      auto sepSize = (range + seperatorCnt - 1) / seperatorCnt;
      SortLocalTaskState& localState = localStates[localStartIndex];
      localState.begin = begin;
      localState.end = end;
      auto& sepVec = localState.localSeperators;
      sepVec.reserve(seperatorCnt);
      for (size_t i = sepSize / 2; i < range; i += sepSize) {
         sepVec.push_back(begin + i);
      }
      trace1.stop();
   }
};

class SortSepSearchTask : public lingodb::scheduler::Task {
   std::vector<uint8_t*>& input;
   size_t splitCnt;
   bool (*compareFn)(uint8_t*, uint8_t*);
   std::vector<SortLocalTaskState>& localStates;
   std::atomic<size_t> startIndex{0};
   std::vector<size_t> workerResvs;
public:
   SortSepSearchTask(std::vector<uint8_t*>& input, size_t splitCnt, bool (*compareFn)(uint8_t*, uint8_t*), std::vector<SortLocalTaskState>& localStates) : input(input), splitCnt(splitCnt), compareFn(compareFn), localStates(localStates) {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         workerResvs.push_back(0);
      }
   }
   bool reserveWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex >= splitCnt) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }
   void consumeWork() override {
      utility::Tracer::Trace trace3(sortSepSearchEvent);
      auto localStartIndex = workerResvs[lingodb::scheduler::currentWorkerId()];
      SortLocalTaskState& localState = localStates[localStartIndex];
      auto segment = std::vector(input.begin() + localState.begin, input.begin() + localState.end);
      for (size_t i = 0; i < localState.globalSeperators.size(); i ++) {
         auto globalSepVal = input[localState.globalSeperators[i]];
         auto it = std::lower_bound(segment.begin(), segment.end(), globalSepVal, compareFn);
         localState.globalSeperators[i] = std::distance(segment.begin(), it) + localState.begin;
      }
      trace3.stop();
   }
};

class SortMergeTask : public lingodb::scheduler::Task {
   std::vector<uint8_t*>& input;
   size_t seperatorCnt;
   bool (*compareFn)(uint8_t*, uint8_t*);
   std::vector<SortLocalTaskState>& localStates;
   uint8_t* output;
   std::vector<size_t>& outputRanges;
   size_t typeSize;
   std::atomic<size_t> startIndex{0};
   std::vector<size_t> workerResvs;

   class MergeSource {
   public:
      std::vector<uint8_t*>& input;
      size_t begin;
      size_t end;
      size_t cursor;

      bool hasNext() {
         return cursor < end;
      }

      uint8_t* next() {
         return input[cursor++];
      }
   };

   struct HeapDatum {
      uint8_t* datum;
      size_t srcIdx;
   };
   typedef struct HeapDatum HeapDatum;

public:
   SortMergeTask(std::vector<uint8_t*>& input, size_t seperatorCnt, bool (*compareFn)(uint8_t*, uint8_t*), std::vector<SortLocalTaskState> &localStates, uint8_t* output, std::vector<size_t>& outputRanges, size_t typeSize) : input(input), seperatorCnt(seperatorCnt), compareFn(compareFn), localStates(localStates), output(output), outputRanges(outputRanges), typeSize(typeSize) {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         workerResvs.push_back(0);
      }
   }
   bool reserveWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex > seperatorCnt) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }

   void consumeWork() override {
      utility::Tracer::Trace trace5(sortMergeEvent);
      auto localStartIndex = workerResvs[lingodb::scheduler::currentWorkerId()];
      std::vector<MergeSource> srcs;
      size_t cnt = 0;
      for (size_t i = 0; i < localStates.size(); i++) {
         size_t begin = localStates[i].begin;
         size_t end = localStates[i].end;
         if (localStartIndex != 0) {
            begin = localStates[i].globalSeperators[localStartIndex-1];
         }
         if (localStartIndex != seperatorCnt) {
            end = localStates[i].globalSeperators[localStartIndex];
         }
         srcs.push_back(MergeSource(input, begin, end, begin));
         cnt += end - begin;
      }

      auto cmp = [&](HeapDatum l, HeapDatum r) {
         return compareFn(l.datum, r.datum);
      };
      std::priority_queue<HeapDatum, std::vector<HeapDatum>, std::function<bool(HeapDatum, HeapDatum)>> minHeap(cmp);
      for (size_t i = 0; i < srcs.size(); i ++) {
         minHeap.push(HeapDatum(srcs[i].next(), i));
      }

      uint8_t* cursor = output+outputRanges[localStartIndex]*typeSize;
      assert(cnt == (outputRanges[localStartIndex+1] - outputRanges[localStartIndex]));
      while (minHeap.size() > 0) {
         auto top = minHeap.top();
         minHeap.pop();
         memcpy(cursor, top.datum, typeSize);
         cursor += typeSize;
         auto& src = srcs[top.srcIdx];
         if (src.hasNext()) {
            minHeap.push(HeapDatum(src.next(), top.srcIdx));
         }
      }
      trace5.stop();
   }
};

class DefaultAllocator : public lingodb::runtime::GrowingBufferAllocator {
   public:
   lingodb::runtime::GrowingBuffer* create(lingodb::runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) override {
      utility::Tracer::Trace trace(createEvent);
      auto* res = new lingodb::runtime::GrowingBuffer(initialCapacity, sizeOfType);
      executionContext->registerState({res, [](void* ptr) { delete reinterpret_cast<lingodb::runtime::GrowingBuffer*>(ptr); }});
      trace.stop();
      return res;
   }
};

class GroupAllocator : public lingodb::runtime::GrowingBufferAllocator {
   std::vector<lingodb::runtime::GrowingBuffer*> buffers;

   public:
   lingodb::runtime::GrowingBuffer* create(lingodb::runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) override {
      auto* res = new lingodb::runtime::GrowingBuffer(initialCapacity, sizeOfType);
      buffers.push_back(res);
      return res;
   }
   ~GroupAllocator() {
      for (auto* buf : buffers) {
         delete buf;
      }
   }
};

} // end namespace

lingodb::runtime::GrowingBuffer* lingodb::runtime::GrowingBuffer::create(lingodb::runtime::GrowingBufferAllocator* allocator, lingodb::runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) {
   return allocator->create(executionContext, sizeOfType, initialCapacity);
}

uint8_t* lingodb::runtime::GrowingBuffer::insert() {
   return values.insert();
}
size_t lingodb::runtime::GrowingBuffer::getLen() const {
   return values.getLen();
}

size_t lingodb::runtime::GrowingBuffer::getTypeSize() const {
   return values.getTypeSize();
}
lingodb::runtime::Buffer lingodb::runtime::GrowingBuffer::sort(lingodb::runtime::ExecutionContext* executionContext, bool (*compareFn)(uint8_t*, uint8_t*)) {
   //todo: make sorting parallel again
   utility::Tracer::Trace trace(sortEvent);
   std::vector<uint8_t*> toSort;
   // TODO parallel copy pointer
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });

   // Step 1
   // TODO splitCnt, seperatorCnt value depend on values.getLen()
   size_t splitCnt = 15;
   size_t seperatorCnt = 15;
   std::vector<SortLocalTaskState> localStates;
   lingodb::scheduler::awaitChildTask(std::make_unique<SortLocalTask>(toSort, seperatorCnt, splitCnt, compareFn, localStates));

   // Step 2
   utility::Tracer::Trace trace2(sortSepSyncEvent);
   std::vector<size_t> samePosSeps;
   samePosSeps.reserve(splitCnt);
   for (size_t i = 0; i < splitCnt; i ++) {
      samePosSeps.push_back(0);
   }
   for (size_t i = 0; i < seperatorCnt; i ++) {
      for (size_t j = 0; j < splitCnt; j ++) {
         samePosSeps[j] = localStates[j].localSeperators[i];
      }
      // TODO USE MEDIAN OF MEDIAN alg
      std::sort(samePosSeps.begin(), samePosSeps.end(), [&](size_t l, size_t r){
         return compareFn(toSort[l], toSort[r]);
      });
      auto globalSep = samePosSeps[samePosSeps.size()/2];
      for (size_t j = 0; j < splitCnt; j ++) {
         localStates[j].globalSeperators.push_back(globalSep);
      }
   }
   trace2.stop();

   // Step 3
   lingodb::scheduler::awaitChildTask(std::make_unique<SortSepSearchTask>(toSort, splitCnt, compareFn, localStates));

   // Step 4
   utility::Tracer::Trace trace4(sortOutputRangeEvent);
   std::vector<size_t> outputRanges;
   outputRanges.push_back(0);
   size_t growingSpan = 0;
   for (size_t j = 0; j <= seperatorCnt; j ++) {
      for (size_t i = 0; i < splitCnt; i ++) {
         size_t begin = localStates[i].begin;
         size_t end = localStates[i].end;
         if (j != 0) {
            begin = localStates[i].globalSeperators[j-1];
         }
         if (j != seperatorCnt) {
            end = localStates[i].globalSeperators[j];
         }
         growingSpan += end - begin;

      }
      outputRanges.push_back(growingSpan);
   }
   outputRanges.push_back(toSort.size());
   trace4.stop();

   // Step 5
   size_t typeSize = values.getTypeSize();
   size_t len = values.getLen();
   uint8_t* sorted = new uint8_t[typeSize * len];
   executionContext->registerState({sorted, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   lingodb::scheduler::awaitChildTask(std::make_unique<SortMergeTask>(toSort, seperatorCnt, compareFn, localStates, sorted, outputRanges, typeSize));
   
   return Buffer{typeSize * len, sorted};
}
lingodb::runtime::Buffer lingodb::runtime::GrowingBuffer::asContinuous(lingodb::runtime::ExecutionContext* executionContext) {
   //todo make more performant...
   std::vector<uint8_t*> toSort;
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });
   size_t typeSize = values.getTypeSize();
   size_t len = values.getLen();
   uint8_t* continuous = new uint8_t[typeSize * len];
   executionContext->registerState({continuous, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = continuous + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }
   return Buffer{typeSize * len, continuous};
}
void lingodb::runtime::GrowingBuffer::destroy(GrowingBuffer* vec) {
   delete vec;
}

lingodb::runtime::GrowingBuffer* lingodb::runtime::GrowingBuffer::merge(lingodb::runtime::ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(mergeEvent);
   GrowingBuffer* first = nullptr;
   for (auto* current : threadLocal->getThreadLocalValues<GrowingBuffer>()) {
      if(!current) continue;
      if (!first) {
         first = current;
      } else {
         first->values.merge(current->values); //todo: cleanup
      }
   }
   trace.stop();
   return first;
}
lingodb::runtime::BufferIterator* lingodb::runtime::GrowingBuffer::createIterator() {
   return values.createIterator();
}

lingodb::runtime::Buffer lingodb::runtime::Buffer::createZeroed(lingodb::runtime::ExecutionContext* executionContext, size_t bytes) {
   auto* ptr = FixedSizedBuffer<uint8_t>::createZeroed(bytes);
   executionContext->registerState({ptr, [bytes](void* ptr) { FixedSizedBuffer<uint8_t>::deallocate((uint8_t*) ptr, bytes); }});
   return Buffer{bytes, ptr};
}

lingodb::runtime::GrowingBufferAllocator* lingodb::runtime::GrowingBufferAllocator::getDefaultAllocator() {
   static DefaultAllocator defaultAllocator;
   return &defaultAllocator;
}

lingodb::runtime::GrowingBufferAllocator* lingodb::runtime::GrowingBufferAllocator::getGroupAllocator(lingodb::runtime::ExecutionContext* executionContext, size_t groupId) {
   auto& state = executionContext->getAllocator(groupId);
   if (state.ptr) {
      return static_cast<GrowingBufferAllocator*>(state.ptr);
   } else {
      auto* newAllocator = new GroupAllocator;
      state.ptr = newAllocator;
      state.freeFn = [](void* ptr) { delete static_cast<GroupAllocator*>(ptr); };
      return newAllocator;
   }
}