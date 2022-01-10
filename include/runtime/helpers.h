#ifndef RUNTIME_HELPERS_H
#define RUNTIME_HELPERS_H
#include "string.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#define EXPORT extern "C" __attribute__((visibility("default")))
#define INLINE __attribute__((always_inline))
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
   static constexpr uint32_t SHORT_LEN = 12;
   uint32_t len;
   uint8_t bytes[SHORT_LEN];

   public:
   VarLen32(uint8_t* ptr, uint32_t len) : len(len) {
      //todo: optimize
      memcpy(bytes, ptr, std::min(len, SHORT_LEN));
      //todo set remaining to zero
      if(len>SHORT_LEN){
         uint8_t** ptrloc=reinterpret_cast<uint8_t**>((&bytes[4]));
         *ptrloc=ptr;
      }
   }
   uint8_t* getPtr() {
      if (len <= SHORT_LEN) {
         return bytes;
      } else {
         return reinterpret_cast<uint8_t*>(*(uintptr_t*) (&bytes[4]));
      }
   }
   char* data() {
      return (char*) getPtr();
   }
   uint32_t getLen() {
      return len;
   }

   __int128 asI128(){
      return *(reinterpret_cast<__int128*>(this));
   }

   operator std::string() { return std::string((char*) getPtr(), len); }
   std::string str() { return std::string((char*) getPtr(), len); }
};
} // end namespace runtime
#endif // RUNTIME_HELPERS_H
