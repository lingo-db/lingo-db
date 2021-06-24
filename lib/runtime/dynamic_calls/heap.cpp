#include "runtime/helpers.h"
#include "string.h"
#include <algorithm>
#include <iostream>
#include <vector>
struct HeapEntry {
   uint8_t* data = nullptr;
   std::vector<uint8_t*> varLen;

   ~HeapEntry(){
      delete[] data;
      for(auto *d:varLen){
         delete [] d;
      }
   }
};
struct Heap {
   bool (*compareFn)(uint8_t*, uint8_t*, int64_t, uint8_t*, uint8_t*, int64_t);
   std::vector<HeapEntry*> heap;
   size_t objSize;
   Heap(size_t max_rows, size_t objSize, bool (*compareFn)(uint8_t*, uint8_t*, int64_t, uint8_t*, uint8_t*, int64_t)) : compareFn(compareFn),objSize(objSize), max_rows(max_rows){
      curr = new HeapEntry;
   }
   HeapEntry* curr;
   size_t max_rows;

   void insert(uint8_t* data) {
      std::uint8_t* clonedData = new uint8_t[objSize];
      memcpy(clonedData, data, objSize);
      curr->data = clonedData;
      heap.push_back(curr);
      curr = new HeapEntry;
      finish_insert();
   }
   //taken from noisepage
   void finish_insert() {
      // If the number of buffered tuples is less than top_k, we're done.
      if (heap.size() < max_rows) {
         return;
      }

      // If we've buffered k elements, build the heap. Note: this is only ever
      // triggered once!
      if (heap.size() == max_rows) {
         buildHeap();
         return;
      }

      // We've buffered ONE more tuple than should be in the top-k, so we may need
      // to reorder the heap. Check if the most recently inserted tuple belongs in
      // the heap.

      HeapEntry* last_insert = heap.back();
      heap.pop_back();

      HeapEntry* heap_top = heap.front();

      if (cmp_fn_(last_insert, heap_top) <= 0) {
         // The last insertion belongs in the top-k. Swap it with the current maximum
         // and sift it down.
         delete heap.front();
         heap.front() = last_insert;
         siftDown();
      }
   }

   int32_t cmp_fn_(const HeapEntry* left, const HeapEntry* right) {
      return compareFn(nullptr, left->data, 0, nullptr, right->data, 0)? -1:1;
   }
   void buildHeap() {
      const auto compare = [this](const HeapEntry* left, const HeapEntry* right) { return cmp_fn_(left, right) < 0; };
      std::make_heap(heap.begin(), heap.end(), compare);
   }

   void siftDown() {
      const uint64_t size = heap.size();
      uint32_t idx = 0;

      HeapEntry* top = heap[idx];

      while (true) {
         uint32_t child = (2 * idx) + 1;

         if (child >= size) {
            break;
         }

         if (child + 1 < size && cmp_fn_(heap[child], heap[child + 1]) < 0) {
            child++;
         }

         if (cmp_fn_(top, heap[child]) >= 0) {
            break;
         }

         std::swap(heap[idx], heap[child]);
         idx = child;
      }

      heap[idx] = top;
   }

   runtime::ByteRange addVarLen(runtime::ByteRange byteRange) {
      uint8_t* ptr = new uint8_t[byteRange.getSize()];
      memcpy(ptr,byteRange.getPtr(),byteRange.getSize());
      curr->varLen.push_back(ptr);
      return {ptr, byteRange.getSize()};
   }
};
EXPORT runtime::Pointer<Heap> _mlir_ciface_topk_builder_create(size_t objSize, size_t maxRows, bool (*fun_ptr)(uint8_t*, uint8_t*, int64_t, uint8_t*, uint8_t*, int64_t)) {// NOLINT (clang-diagnostic-return-type-c-linkage)
   return new Heap(maxRows, objSize, fun_ptr);
}

EXPORT runtime::ByteRange _mlir_ciface_topk_builder_add_var_len(runtime::Pointer<Heap>* builder, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->addVarLen(*data);
}
EXPORT runtime::Pair<bool, runtime::ByteRange> _mlir_ciface_topk_builder_add_nullable_var_len(runtime::Pointer<Heap>* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::ByteRange(nullptr, 0)};
   }
   return {false, (*builder)->addVarLen(*data)};
}

EXPORT void _mlir_ciface_topk_builder_merge(runtime::Pointer<Heap>* heap, runtime::Pointer<uint8_t>* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (*heap)->insert((*data).get());
}
EXPORT runtime::Pointer<Heap> _mlir_ciface_topk_builder_build(runtime::Pointer<Heap>* heap) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return *heap;
}
EXPORT size_t _mlir_ciface_topk_entries(runtime::Pointer<Heap>* heap){// NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*heap)->heap.size();
}
EXPORT runtime::Pointer<uint8_t> _mlir_ciface_topk_get_entry(runtime::Pointer<Heap>* heap, size_t i){// NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*heap)->heap[(*heap)->heap.size()-1-i]->data;
}