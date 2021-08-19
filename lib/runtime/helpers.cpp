#include "runtime/helpers.h"
uint8_t* runtime::ObjectBufferStorage::alloc(){
   auto* currPart = parts[parts.size() - 1];
   if (!currPart->fits(objSize)) {
      size_t newSize = currPart->getCapacity() * 2;
      parts.push_back(new Part(newSize));
      currPart = parts[parts.size() - 1];
   }
   return currPart->alloc(objSize);
}