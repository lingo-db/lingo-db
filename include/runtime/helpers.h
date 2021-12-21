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

class VarLen32 {
   uint32_t len;
   uint8_t type;
   uint8_t bytes[11];

   public:
   uint8_t* getPtr() {
      if (len <= 11) {
         return bytes;
      } else {
         return reinterpret_cast<uint8_t*>(*(uintptr_t*) (&bytes[3]));
      }
   }
   char* data() {
      return (char*) getPtr();
   }
   uint32_t getLen() {
      return len;
   }
   uint8_t getType() {
      return type;
   }
   operator std::string() { return std::string((char*) getPtr(), len); }
   std::string str() { return std::string((char*) getPtr(), len); }
};
} // end namespace runtime
#endif // RUNTIME_HELPERS_H
