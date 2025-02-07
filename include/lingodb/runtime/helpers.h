#ifndef LINGODB_RUNTIME_HELPERS_H
#define LINGODB_RUNTIME_HELPERS_H
#include "string.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <sys/mman.h>

#define EXPORT extern "C" __attribute__((visibility("default")))
#define INLINE __attribute__((always_inline))
namespace lingodb::runtime {

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
#if defined(ASAN_ACTIVE)
   uint64_t x=0;
   memcpy(&x,p,len);
   return x;
#else
   if (len >= 8) return unalignedLoad64(p); //best case
   /*uint64_t x;
   memcpy(&x,p,len);
   return x;
   */
   if (cachelineRemains8(p)) {
      auto bytes = unalignedLoad64(p);
      auto shift = len * 8;
      auto mask = ~((~0ull) << shift);
      auto ret = bytes & mask; //we can load bytes, but have to shift for invalid bytes
      return ret;
   }
   return unalignedLoad64(p + len - 8) >> (64 - len * 8);
#endif
}

class VarLen32 {
   private:
   uint32_t len;

   public:
   static constexpr uint32_t shortLen = 12;
   union {
      uint8_t bytes[shortLen];
      struct __attribute__((__packed__)) {
         uint32_t first4;
         uint64_t last8;
      };
   };

   private:
   void storePtr(const uint8_t* ptr) {
      const uint8_t** ptrloc = reinterpret_cast<const uint8_t**>((&bytes[4]));
      *ptrloc = ptr;
   }

   public:
   static VarLen32 fromString(std::string str) {
      if (str.size() <= shortLen) {
         return VarLen32(reinterpret_cast<const uint8_t*>(str.data()), str.size());
      }
      auto* ptr = new uint8_t[str.size()];
      memcpy(ptr, str.data(), str.size());
      return VarLen32(ptr, str.size());
   }
   VarLen32() : VarLen32(nullptr, 0) {}
   VarLen32(const uint8_t* ptr, uint32_t len) : len(len) {
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
   bool isShort() {
      return len <= shortLen;
   }
   uint32_t getLen() {
      return len;
   }

   __int128 asI128() {
      return *(reinterpret_cast<__int128*>(this));
   }

   operator std::string() { return std::string((char*) getPtr(), getLen()); }
   std::string str() { return std::string((char*) getPtr(), getLen()); }
};

template <class T>
struct LegacyFixedSizedBuffer {
   LegacyFixedSizedBuffer(size_t size) : ptr((T*) malloc(size * sizeof(T))) {
      runtime::MemoryHelper::zero((uint8_t*) ptr, size * sizeof(T));
   }
   T* ptr;
   void setNewSize(size_t newSize) {
      free(ptr);
      ptr = (T*) malloc(newSize * sizeof(T));
      runtime::MemoryHelper::zero((uint8_t*) ptr, newSize * sizeof(T));
   }
   T& at(size_t i) {
      return ptr[i];
   }
   T* getPtr(size_t i) {
      return &ptr[i];
   }
   ~LegacyFixedSizedBuffer() {
      free(ptr);
   }
};

template <class T>
struct FixedSizedBuffer {
   static bool shouldUseMMAP(size_t elements) {
      return (sizeof(T) * elements) >= 65536;
   }
   static T* createZeroed(size_t elements) {
      if (shouldUseMMAP(elements)) {
         return (T*) mmap(NULL, elements * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
      } else {
         auto res = (T*) malloc(elements * sizeof(T));
         runtime::MemoryHelper::zero((uint8_t*) res, elements * sizeof(T));
         return res;
      }
   }
   static void deallocate(T* ptr, size_t elements) {
      if (shouldUseMMAP(elements)) {
         munmap(ptr, elements * sizeof(T));
      } else {
         free(ptr);
      }
   }
};
template <typename T>
T* tag(T* ptr, T* previousPtr, size_t hash) {
   constexpr uint64_t ptrMask = 0x0000ffffffffffffull;
   constexpr uint64_t tagMask = 0xffff000000000000ull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   size_t previousAsInt = reinterpret_cast<size_t>(previousPtr);
   size_t previousTag = previousAsInt & tagMask;
   size_t currentTag = hash & tagMask;
   auto tagged = (asInt & ptrMask) | previousTag | currentTag;
   auto* res = reinterpret_cast<T*>(tagged);
   return res;
}
template <typename T>
T* filterTagged(T* ptr, size_t hash) {
   constexpr uint64_t ptrMask = 0x0000ffffffffffffull;
   constexpr uint64_t tagMask = 0xffff000000000000ull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   size_t requiredTag = hash & tagMask;
   size_t currentTag = hash & tagMask;
   return ((currentTag | requiredTag) == currentTag) ? reinterpret_cast<T*>(asInt & ptrMask) : nullptr;
}
template <typename T>
T* untag(T* ptr) {
   constexpr size_t ptrMask = 0x0000ffffffffffffull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   return reinterpret_cast<T*>(asInt & ptrMask);
}
} // end namespace lingodb::runtime
#endif // LINGODB_RUNTIME_HELPERS_H
