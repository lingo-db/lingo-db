#ifndef RUNTIME_HELPERS_H
#define RUNTIME_HELPERS_H
#include "string.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#define EXPORT extern "C" __attribute__((visibility("default")))

namespace runtime {
class Str {
   char* pointer;
   size_t size;

   public:
   Str(char* ptr, size_t len) : pointer(ptr), size(len) {}
   operator std::string() { return std::string(pointer, size); }
   std::string str() { return std::string(pointer, size); }
   size_t len() {
      return size;
   }
   char* data() {
      return pointer;
   }
};
class Bytes {
   uint8_t* pointer;
   size_t size;

   public:
   Bytes(uint8_t* ptr, size_t bytes) : pointer(ptr), size(bytes) {}
   uint8_t* getPtr() {
      return pointer;
   }
   size_t getSize() {
      return size;
   }
};
} // end namespace runtime
#endif // RUNTIME_HELPERS_H
