#include "runtime/helpers.h"
#include "string.h"
#include <algorithm>
#include <iostream>
#include <vector>
struct Vector {
   runtime::ByteRange values;
   runtime::ByteRange varLenData;
};
EXPORT void _mlir_ciface_sort(runtime::Pointer<Vector>* vector, size_t objSize, bool (*fun_ptr)(uint8_t*, uint8_t*, int64_t, uint8_t*, uint8_t*, int64_t, uint8_t*, uint8_t*, int64_t), runtime::Pointer<Vector>* res) {
   std::vector<uint8_t*> toSort;

   uint8_t* valuePtr = (*vector)->values.getPtr();
   uint8_t* varLenPtr = (*vector)->varLenData.getPtr();
   size_t bytes = (*vector)->values.getSize();
   for (auto* i = valuePtr; i < valuePtr + bytes; i += objSize) {
      toSort.push_back(i);
   }
   uint8_t* nullPtr = nullptr;
   std::sort(toSort.begin(), toSort.end(), [&](uint8_t* left, uint8_t* right) {
      return fun_ptr(nullPtr, varLenPtr, 0, nullPtr, left, 0, nullPtr, right, 0);
   });
   uint8_t* sortedValues = new uint8_t[bytes];
   for (size_t i = 0; i < bytes / objSize; i++) {
      uint8_t* ptr = sortedValues + (i * objSize);
      memcpy(ptr, toSort[i], objSize);
   }
   size_t varLenBytes = (*vector)->varLenData.getSize();
   uint8_t* newVarLenData = new uint8_t[varLenBytes];
   memcpy(newVarLenData, (*vector)->varLenData.getPtr(), varLenBytes);

   (*res).ref() = {{sortedValues, bytes}, {newVarLenData, varLenBytes}};
}