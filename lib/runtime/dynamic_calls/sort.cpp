#include "runtime/helpers.h"
#include "string.h"
#include <algorithm>
#include <iostream>
#include <vector>

EXPORT void rt_sort(size_t len,uint8_t* valuePtr, size_t objSize, bool (*fun_ptr)(uint8_t*, uint8_t*)) {
   std::vector<uint8_t*> toSort;

   size_t bytes =len*objSize;
   for (auto* i = valuePtr; i < valuePtr + bytes; i += objSize) {
      toSort.push_back(i);
   }
   std::sort(toSort.begin(), toSort.end(), [&](uint8_t* left, uint8_t* right) {
      return fun_ptr(left,right);
   });
   uint8_t* sorted=new uint8_t[bytes];
   for (size_t i = 0; i < bytes / objSize; i++) {
      uint8_t* ptr = sorted + (i * objSize);
      memcpy(ptr, toSort[i], objSize);
   }
   memcpy(valuePtr,sorted,bytes);
   delete[] sorted;
}