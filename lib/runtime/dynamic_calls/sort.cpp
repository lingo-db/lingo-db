#include "runtime/helpers.h"
#include "string.h"
#include <algorithm>
#include <iostream>
#include <vector>

EXPORT void _mlir_ciface_sort(runtime::Pointer<runtime::Vector>* vector, size_t objSize, bool (*fun_ptr)(uint8_t*, uint8_t*, int64_t, uint8_t*, uint8_t*, int64_t)) {
   std::vector<uint8_t*> toSort;

   uint8_t* valuePtr = &(*vector)->values[0];
   size_t bytes = (*vector)->values.size();
   for (auto* i = valuePtr; i < valuePtr + bytes; i += objSize) {
      toSort.push_back(i);
   }
   uint8_t* nullPtr = nullptr;
   std::sort(toSort.begin(), toSort.end(), [&](uint8_t* left, uint8_t* right) {
      return fun_ptr(nullPtr, left, 0, nullPtr, right, 0);
   });
   std::vector<uint8_t> sortedValues((*vector)->values.size());
   for (size_t i = 0; i < bytes / objSize; i++) {
      uint8_t* ptr = &sortedValues[0] + (i * objSize);
      memcpy(ptr, toSort[i], objSize);
   }
   (*vector)->values=sortedValues;
}