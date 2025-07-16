#ifndef LINGODB_RUNTIME_LISTRUNTIME_H
#define LINGODB_RUNTIME_LISTRUNTIME_H
#include "lingodb/runtime/Buffer.h"

#include <lingodb/runtime/helpers.h>

namespace lingodb::runtime {
class List {
   size_t sizeOfType;
   size_t len;
   std::vector<uint8_t> values;
   List(size_t sizeOfType) : sizeOfType(sizeOfType), len(), values(2 * sizeOfType) {}

   public:
   static List* create(size_t sizeOfType);
   uint8_t* at(size_t pos);
   static List* fromBuffer(size_t sizeOfType, Buffer buffer);
   uint8_t* append();
   size_t size();
   Buffer getBuffer();
};

}; // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_LISTRUNTIME_H