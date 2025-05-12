#ifndef LINGODB_RUNTIME_PERFECTHASHTABLE_H
#define LINGODB_RUNTIME_PERFECTHASHTABLE_H
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/StringRuntime.h"
#include <cstring>

namespace lingodb::runtime {
class GrowingBuffer;
struct HashParams {
   uint32_t a;
   uint32_t b;
};

// TODO RENAME
struct LKEntry {
   bool empty;
   uint64_t hashvalue;
   VarLen32 key;
};

class PerfectHashView {
   // Hash function parameters
   // HashParams auxHashParams[2];  // Parameters for auxiliary hash functions
   // int8_t* g;                  // Displacement values
   // lingodb::runtime::VarLen32** lookupTable;
   // size_t tableSize;                       // Size of the hash table (number of keys)
   // size_t r;                               // Range for intermediate hash values
public:
   LKEntry* lookupTable;
   std::vector<size_t> g;                  // Displacement values
   HashParams auxHashParams[2];  // Parameters for auxiliary hash functions
   // Size of the hash table (number of keys). max 256
   uint8_t tableSize;
   // Size of displacement table. max 65536
   uint16_t r;
   
   std::vector<std::optional<std::string>> lookupTableRaw;

   // Constructor
   PerfectHashView(const std::vector<std::string>& keySet) {
      tableSize = keySet.size() * 2;
      if (tableSize > 0) {
         lookupTable = (LKEntry*) malloc(sizeof(LKEntry) * tableSize);
         lookupTableRaw.resize(tableSize);
      }
   }

   // Build the perfect hash function
   static PerfectHashView* buildPerfectHash(const std::vector<std::string>& keySet);

   static PerfectHashView* build(FlexibleBuffer* lkbuffer, FlexibleBuffer* gvalues);

   // Universal hash function: h(x) = ((a*x + b) mod p) mod r
   size_t universalHash(const char* keyPtr, size_t keyLen, const HashParams& params, bool trim) const {
      size_t hash = 0;
      const uint32_t prime = 0x7FFFFFFF; // 2^31 - 1

      // // 4 bytes as a unit to compute hash
      int i = 0;
      for (; i < keyLen - 4; i += 4) {
         uint32_t c;
         std::memcpy(&c, keyPtr+i, sizeof(uint32_t));
         hash = (hash * params.a + c) & prime;
      }

      // // deal with not mutiply of 4 part
      size_t restLen = keyLen - i;
      if (restLen == 3) {
         uint32_t c;
         std::memcpy(&c, keyPtr+i, sizeof(uint32_t));
         c &= 0x00FFFFFF;
         hash = (hash * params.a + c) & prime;
      } else if (restLen == 2) {
         uint16_t c;
         std::memcpy(&c, keyPtr+i, sizeof(uint16_t));
         hash = (hash * params.a + c) & prime;
         restLen -= 2;
      } else if (restLen == 1) {
         uint8_t c = static_cast<uint8_t>(*(keyPtr+restLen));
         hash = (hash * params.a + c) & prime;
      }

      hash = (hash * params.b) & prime;
      if  (trim) {
         return hash % this->r;
      }
      return hash;
   }

   size_t computeHash(uint8_t* keyPtr) {
      // printf("~~~ computeHash %p\n", keyPtr);
      // return 0;
      if (keyPtr == nullptr) {
         printf("~~~!! computeHash %p\n", keyPtr);
      }
      // printf("~~~ computeHash %s %lu\n", key.str().c_str(), h);

      // lingodb::runtime::VarLen32 key;
      // std::memcpy(&key, keyPtr, sizeof(key));
      lingodb::runtime::VarLen32& key = *(reinterpret_cast<lingodb::runtime::VarLen32*>(keyPtr));
      size_t h1 = universalHash(key.data(), key.getLen(), auxHashParams[0], false);
      size_t h2 = universalHash(key.data(), key.getLen(), auxHashParams[1], true);

      // static bool flag;
      // if (key.str() == "Clerk#000000536" && !flag) {
      //    flag = true;
      //    size_t h = h1 + g[h2];
      //    size_t idx = h % tableSize;
      //    auto entry = lookupTable[idx];
      //    printf("### key %lu %lu %lu entry: %lu\n", h, tableSize, idx, entry.hashvalue);

      //    printf("aux %u %u %u %u \n", auxHashParams[0].a, auxHashParams[0].b, auxHashParams[1].a, auxHashParams[1].b);
      // }

      return h1 + g[h2];
   }

   // Check if a key exists in the original key set
   void* containHash(size_t h) {
      size_t idx = h % tableSize;
      
      // 直接检查查找表中的值是否匹配
      // TODO DELETE contains. LOOKUP logics in MLIR
      return &lookupTable[idx];
   }

   void* dryRun() {
      static lingodb::runtime::VarLen32 dryRunkey;
      if (dryRunkey.getLen() == 0) {
         dryRunkey = lingodb::runtime::VarLen32::fromString("Clerk#000000536");
      }
      size_t h1 = universalHash(dryRunkey.data(), dryRunkey.getLen(), auxHashParams[0], false);
      size_t h2 = universalHash(dryRunkey.data(), dryRunkey.getLen(), auxHashParams[1], true);

      size_t h = h1 + g[h2];

      static LKEntry* first;
      static LKEntry* second;
      if (!first) {
         for (int i = 0; i < tableSize; i ++) {
            if (!lookupTable[i].empty) {
               if (!first) first = &lookupTable[i];
               else {
                  printf("<<<< dryRun %d\n", i);
                  second = &lookupTable[i];
                  break;
               }
            }
         }
      }

      // if (h % 50 != 0) {
      //    return second;
      // }
      return first;
   }

   size_t dryRunHash() {
      static LKEntry* first;
      if (!first) {
         for (int i = 0; i < tableSize; i ++) {
            if (!lookupTable[i].empty) {
               first = &lookupTable[i];
               printf("<<<< dryRunHash %d\n", i);
               break;
            }
         }
      }
      return first->hashvalue;
   }

   // size_t computeHash(lingodb::runtime::VarLen32 key) {
   //    // printf("~~~ computeHash %p\n", keyPtr);
   //    // return 0;
   //    size_t h1 = universalHash(key, auxHashParams[0], false);
   //    size_t h2 = universalHash(key, auxHashParams[1], true);

   //    return (h1 + g[h2]);
   // }

   // Get the number of keys in the hash table
   size_t size() const {
      return tableSize;
   }

   // TODO
   // // Display statistics about the hash function
   // void printStats() const {
   //    std::cout << "FCH Perfect Hash Statistics:" << std::endl;
   //    std::cout << "Number of keys: " << tableSize << std::endl;
   //    std::cout << "Displacement table size: " << g.size() << std::endl;
   //    std::cout << "Bits per key: " << (float)(g.size() * 8) / tableSize << std::endl;
   // }
};
} // end namespace lingodb::runtime
#endif // LINGODB_RUNTIME_PERFECTHASHTABLE_H
