#include "lingodb/runtime/Sorting.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"
#include "lingodb/utility/Tracer.h"
#include <queue>

namespace {
static utility::Tracer::Event sortAllocEvent("GrowingBuffer", "sortAlloc");
static utility::Tracer::Event sortCopyEvent("GrowingBuffer", "sortCopy");
static utility::Tracer::Event sortLocalEvent("GrowingBuffer", "sortLocal");
static utility::Tracer::Event sortSepSearchEvent("GrowingBuffer", "sortSepSearch");
static utility::Tracer::Event sortOutputRangeEvent("GrowingBuffer", "sortOutputRange");
static utility::Tracer::Event sortMergeEvent("GrowingBuffer", "sortMerge");

struct SortSplitState {
   size_t begin{0};
   size_t end{0};
   std::vector<size_t> localSeperators;
   // ideally we only need to keep one vector of seperators that local/global seperators reuse it.
   // but we keep two vector for case of debug reason.
   std::vector<size_t> globalSeperators;
};

struct SortContext {
   std::vector<uint8_t*>& input;
   size_t splitCnt;
   size_t seperatorCnt;
   size_t typeSize;
   bool (*compareFn)(uint8_t*, uint8_t*);
   std::vector<SortSplitState>& localStates;
};

// This task copy a vector of buffer to a single big vector. Every worker takes buffer and do parallel copy
class SortCopyTask : public lingodb::scheduler::TaskWithImplicitContext {
   const std::vector<lingodb::runtime::Buffer>& buffers;
   std::vector<uint8_t*>& copy;
   size_t typeSize;
   std::atomic<size_t> startIndex{0};
   std::vector<size_t> workerResvs;
   std::vector<size_t> bufferOffsets;
public:
   SortCopyTask(const std::vector<lingodb::runtime::Buffer>& buffers, std::vector<uint8_t*>& copy, size_t typeSize) : buffers(buffers), copy(copy), typeSize(typeSize) {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         workerResvs.push_back(0);
      }
      size_t cnt = 0;
      for (size_t i = 0; i < buffers.size(); i ++) {
         bufferOffsets.push_back(cnt);
         cnt += buffers[i].numElements;
      }
   }
   bool allocateWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex >= buffers.size()) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }
   void performWork() override {
      utility::Tracer::Trace trace(sortCopyEvent);
      auto localStartIndex = workerResvs[lingodb::scheduler::currentWorkerId()];
      const lingodb::runtime::Buffer& buffer = buffers[localStartIndex];
      auto offset = bufferOffsets[localStartIndex];
      for (size_t i = 0; i < buffer.numElements; i ++) {
         copy[i+offset] = &buffer.ptr[i * typeSize];
      }
      trace.stop();
   }
};


// A whole input vector is split into several equal sized splits. And every worker in this task 
// 1. take a split and sort it.
// 2. calculate local seperators of this split. for example split bounded by `|`, separator count for 3
// ... | 3, 5, 8, 13, 16, 17, 19, 26, 29, 31, 33, 38, 39 | ...
//             |              |               |
//            sep1           sep2            sep3
// Local seperators are equally distributed throught split.
class SortLocalTask : public lingodb::scheduler::TaskWithImplicitContext {
   SortContext& sctx;
   size_t splitSize;
   std::atomic<size_t> startIndex{0};
   std::vector<size_t> workerResvs;

   public:
   SortLocalTask(SortContext& sctx) : sctx(sctx) {
      auto splitCnt = sctx.splitCnt;
      splitSize = (sctx.input.size()+splitCnt-1)/splitCnt;
      workerResvs.resize(lingodb::scheduler::getNumWorkers());
      sctx.localStates.resize(splitCnt);
   }
   bool allocateWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex >= sctx.splitCnt) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }
   void performWork() override {
      utility::Tracer::Trace trace1(sortLocalEvent);
      auto localStartIndex = workerResvs[lingodb::scheduler::currentWorkerId()];
      auto begin = localStartIndex * splitSize;
      auto end = (localStartIndex + 1) * splitSize;
      auto& input = sctx.input;
      if (end > input.size()) {
         end = input.size();
      }
      std::sort(input.begin() + begin, input.begin() + end, sctx.compareFn);

      auto range = end - begin;
      SortSplitState& localState = sctx.localStates[localStartIndex];
      localState.begin = begin;
      localState.end = end;
      auto seperatorCnt = sctx.seperatorCnt;
      auto startOffset = range / seperatorCnt / 2;
      auto& sepVec = localState.localSeperators;
      sepVec.reserve(seperatorCnt);
      // assert to make sure seperatorCnt * range not overflow. the assertions are loose conditions.
      // should never be asserted fail.
      assert(range < 2000000000);
      assert(seperatorCnt < 2000000);
      for (double i = 0; i < seperatorCnt; i ++) {
         // use `i * range / seperatorCnt` instead of `range / seperatorCnt * i` to generate evenly
         // distributed seperator sequence. and `i * range` is not likely to overflow.
         sepVec.push_back(begin + startOffset + i * range / seperatorCnt);
      }
      trace1.stop();
   }
};

class SortSepSearchTask : public lingodb::scheduler::TaskWithImplicitContext {
   SortContext& sctx;
   std::atomic<size_t> startIndex{0};
   std::vector<size_t> workerResvs;
   std::vector<std::vector<size_t>> workerSamePosSeps;
public:
   SortSepSearchTask(SortContext& sctx) : sctx(sctx) {
      workerResvs.resize(lingodb::scheduler::getNumWorkers());
      workerSamePosSeps.resize(lingodb::scheduler::getNumWorkers());
      for (size_t i = 0; i < sctx.splitCnt; i ++) {
         sctx.localStates[i].globalSeperators = std::vector<size_t>(sctx.seperatorCnt);
      }
   }
   bool allocateWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex >= sctx.seperatorCnt) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }
   void performWork() override {
      utility::Tracer::Trace trace3(sortSepSearchEvent);
      auto workerId = lingodb::scheduler::currentWorkerId();
      auto localStartIndex = workerResvs[workerId];
      auto splitCnt = sctx.splitCnt;
      // samePosSeps is allocted once per worker to reduce unnecessary allocs
      std::vector<size_t>& samePosSeps = workerSamePosSeps[workerId];
      if (samePosSeps.size() == 0) {
         samePosSeps.resize(splitCnt);
      }
      // Step 1. collect local seperators of same position from all splits
      for (size_t j = 0; j < splitCnt; j ++) {
         samePosSeps[j] = sctx.localStates[j].localSeperators[localStartIndex];
      }
      // Step 2. find median of local seperators' element, assign it to `globalSepVal`
      auto& input = sctx.input;
      std::sort(samePosSeps.begin(), samePosSeps.end(), [&](size_t l, size_t r){
         return sctx.compareFn(input[l], input[r]);
      });
      auto globalSepIdx = samePosSeps[samePosSeps.size()/2];
      auto* globalSepVal = input[globalSepIdx];
      // Step 3. take each split as input and do binary search `globalSepVal`.
      for (size_t i = 0; i < splitCnt; i ++) {
         auto &localState = sctx.localStates[i];
         auto segment = std::vector(input.begin() + localState.begin, input.begin() + localState.end);
         // binary search for lower bound if not exactly matched
         auto it = std::lower_bound(segment.begin(), segment.end(), globalSepVal, sctx.compareFn);
         // every `localState.globalSeperators[localStartIndex]` in `localStates` would be aligned. 
         // `localState.globalSeperators[localStartIndex]` then could be input for next merge phase.
         localState.globalSeperators[localStartIndex] = std::distance(segment.begin(), it) + localState.begin;
      }
      trace3.stop();
   }
};

// Merge all the splits into one output. Every worker map two adjacent seperators to all splits and 
// collect result segment. Then every worker merge all segment using minheap, which push and poped 
// one by one into result output.
class SortMergeTask : public lingodb::scheduler::TaskWithImplicitContext {
   SortContext& sctx;
   uint8_t* output;
   std::vector<size_t>& outputRanges;
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

public:
   SortMergeTask(SortContext& sctx, uint8_t* output, std::vector<size_t>& outputRanges) : sctx(sctx), output(output), outputRanges(outputRanges){
      workerResvs.resize(lingodb::scheduler::getNumWorkers());
   }
   bool allocateWork() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex > sctx.seperatorCnt) {
         workExhausted.store(true);
         return false;
      }
      workerResvs[lingodb::scheduler::currentWorkerId()] = localStartIndex;
      return true;
   }

   void performWork() override {
      utility::Tracer::Trace trace5(sortMergeEvent);
      auto localStartIndex = workerResvs[lingodb::scheduler::currentWorkerId()];
      std::vector<MergeSource> srcs;
      size_t cnt = 0;
      auto& localStates = sctx.localStates;
      for (size_t i = 0; i < localStates.size(); i++) {
         size_t begin = localStates[i].begin;
         size_t end = localStates[i].end;
         if (localStartIndex != 0) {
            begin = localStates[i].globalSeperators[localStartIndex-1];
         }
         if (localStartIndex != sctx.seperatorCnt) {
            end = localStates[i].globalSeperators[localStartIndex];
         }
         srcs.push_back(MergeSource(sctx.input, begin, end, begin));
         cnt += end - begin;
      }

      auto cmp = [&](HeapDatum l, HeapDatum r) {
         return !sctx.compareFn(l.datum, r.datum);
      };
      std::priority_queue<HeapDatum, std::vector<HeapDatum>, std::function<bool(HeapDatum, HeapDatum)>> minHeap(cmp);
      for (size_t i = 0; i < srcs.size(); i ++) {
         if (srcs[i].hasNext()) {
            minHeap.push(HeapDatum(srcs[i].next(), i));
         }
      }

      auto typeSize = sctx.typeSize;
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

const size_t minSplitSize = 512;
void calcSplitSeperator(size_t inputSize, size_t workerNum, size_t& splitCnt, size_t& seperatorCnt) {
   size_t sizePerWorker = inputSize / workerNum;
   auto normalizeSepCnt = [&]() {
      size_t sizePerSplit = inputSize / splitCnt;
      size_t minSepSegment = 8;
      if (minSepSegment * seperatorCnt > sizePerSplit) {
         seperatorCnt = sizePerSplit / minSepSegment;
      }
   };

   if (sizePerWorker < minSplitSize*4) {
      // we don't split smaller granularity than 512. over-parallelism on small input may not be a 
      // good idea, since
      // 1. input is small. it should be really fast without using all workers. even if more workers
      //    improve performance, it may have a performance boost from 10 µs to 7 µs, which actually
      //    make no difference.
      // 2. less competing for limited task. by having few workers running on small size input, we
      //    "may" can save resources like lock aquiring, fiber alloc/run cost.
      splitCnt = inputSize / minSplitSize;
      seperatorCnt = workerNum;
      normalizeSepCnt();
      return;
   } else if (sizePerWorker > 32768 * 4) {
      // a fixed relative small number should help local sorter operate sort algorithm entirely on cache,
      // resulting in good performance.
      size_t splitSize = 32768;
      splitCnt = inputSize / splitSize;
      // we are more conservative about give splitCnt a big number. increase seperators number will 
      // increase memory to store those seperators and time spending on find median seperators.
      // a factor 4 is a quite good scale for parallism already.
      seperatorCnt = workerNum * 4;
      normalizeSepCnt();
      return;
   }

   // moderate size of input, sizePerWorker in range [2048, 131072]
   if (sizePerWorker > 65536) {
      splitCnt = workerNum * 4;
   } else {
      splitCnt = workerNum * 2;
   }
   seperatorCnt = workerNum * 2;
   normalizeSepCnt();
}
} //end namespace

namespace lingodb::runtime {
bool canParallelSort(const size_t valueSize) {
   return valueSize > minSplitSize;
}
lingodb::runtime::Buffer parallelSort(lingodb::runtime::FlexibleBuffer& values, bool (*compareFn)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace trace1(sortAllocEvent);
   std::vector<uint8_t*> toSort = std::vector<uint8_t*>(values.getLen());
   trace1.stop();
   lingodb::scheduler::awaitChildTask(std::make_unique<SortCopyTask>(values.getBuffers(), toSort, values.getTypeSize()));

   // Step 1 local sort and calculate local seperators
   size_t splitCnt, seperatorCnt;
   calcSplitSeperator(values.getLen(), lingodb::scheduler::getNumWorkers(), splitCnt, seperatorCnt);
   std::vector<SortSplitState> localStates;
   localStates.reserve(splitCnt);
   SortContext sctx = SortContext(toSort, splitCnt, seperatorCnt, values.getTypeSize(), compareFn, localStates);
   lingodb::scheduler::awaitChildTask(std::make_unique<SortLocalTask>(sctx));

   // Step 2 calculate global seperators
   lingodb::scheduler::awaitChildTask(std::make_unique<SortSepSearchTask>(sctx));

   // Step 3 generate output ranges
   utility::Tracer::Trace trace3(sortOutputRangeEvent);
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
   trace3.stop();

   // Step 4 merge all sorted splits to output
   utility::Tracer::Trace trace4(sortAllocEvent);
   size_t typeSize = values.getTypeSize();
   size_t len = values.getLen();
   uint8_t* sorted = new uint8_t[typeSize * len];
   trace4.stop();
   auto* executionContext= lingodb::runtime::getCurrentExecutionContext();
   executionContext->registerState({sorted, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   lingodb::scheduler::awaitChildTask(std::make_unique<SortMergeTask>(sctx, sorted, outputRanges));

   return lingodb::runtime::Buffer{typeSize * len, sorted};
}
} // end namespace lingodb::runtime