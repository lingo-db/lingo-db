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

struct MemoryHelper {
   static uint8_t* resize(uint8_t* old, size_t oldNumBytes, size_t newNumBytes) {
      uint8_t* newBytes = (uint8_t*) malloc(newNumBytes);
      memcpy(newBytes, old, oldNumBytes);
      free(old);
      return newBytes;
   }
   static void fill(uint8_t* ptr, uint8_t val, size_t size) {
      memset(ptr, val, size);
   }
   static void zero(uint8_t* ptr, size_t size) { fill(ptr, 0, size); }
};

static uint64_t unalignedLoad64(const uint8_t* p) {
   uint64_t result;
   memcpy(&result, p, sizeof(result));
   return result;
}

static uint32_t unalignedLoad32(const uint8_t* p) {
   uint32_t result;
   memcpy(&result, p, sizeof(result));
   return result;
}
static bool cachelineRemains8(const uint8_t* p) {
   return (reinterpret_cast<uintptr_t>(p) & 63) <= 56;
}
static uint64_t read8PadZero(const uint8_t* p, uint32_t len) {
   if (len == 0) return 0; //do not dereference!
   if (len >= 8) return unalignedLoad64(p); //best case
   if (cachelineRemains8(p)) {
      auto bytes = unalignedLoad64(p);
      auto shift = len * 8;
      auto mask = ~((~0ull) << shift);
      auto ret = bytes & mask; //we can load bytes, but have to shift for invalid bytes
      return ret;
   }
   return unalignedLoad64(p + len - 8) >> (64 - len * 8);
}

class VarLen32 {
   private:
   uint32_t len;

   public:
   static constexpr uint32_t shortLen = 12;
   static constexpr uint32_t lazyMask = 0x80000000;
   union {
      uint8_t bytes[shortLen];
      struct __attribute__((__packed__)) {
         uint32_t first4;
         uint64_t last8;
      };
   };

   private:
   void storePtr(uint8_t* ptr) {
      uint8_t** ptrloc = reinterpret_cast<uint8_t**>((&bytes[4]));
      *ptrloc = ptr;
   }

   public:
   VarLen32(uint8_t* ptr, uint32_t len) : len(len) {
      if (len > shortLen) {
         this->first4 = unalignedLoad32(ptr);
         storePtr(ptr);
      } else if (len > 8) {
         this->first4 = unalignedLoad32(ptr);
         this->last8 = unalignedLoad64(ptr + len - 8);
         uint32_t duplicate = 12 - len;
         this->last8 >>= (duplicate * 8);
      } else {
         auto bytes = read8PadZero(ptr, len);
         this->first4 = bytes;
         this->last8 = bytes >> 32;
      }
   }
   uint8_t* getPtr() {
      if (len <= shortLen) {
         return bytes;
      } else {
         return reinterpret_cast<uint8_t*>(*(uintptr_t*) (&bytes[4]));
      }
   }
   char* data() {
      return (char*) getPtr();
   }
   uint32_t getLen() {
      return len & ~lazyMask;
   }
   bool isLazy() {
      return (len & lazyMask) != 0;
   }

   __int128 asI128() {
      return *(reinterpret_cast<__int128*>(this));
   }

   operator std::string() { return std::string((char*) getPtr(), getLen()); }
   std::string str() { return std::string((char*) getPtr(), getLen()); }
};
class X {
   static int add(std::string, int b);
   int mul(int x, int y);
};
} // end namespace runtime
#endif // RUNTIME_HELPERS_H
