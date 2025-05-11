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
   std::vector<LKEntry> lookupTable;
   std::vector<size_t> g;                  // Displacement values
   HashParams auxHashParams[2];  // Parameters for auxiliary hash functions
   // Size of the hash table (number of keys). max 256
   uint8_t tableSize;
   // Size of displacement table. max 65536
   uint16_t r;
   
   std::vector<std::optional<std::string>> lookupTableRaw;

   // Universal hash function: h(x) = ((a*x + b) mod p) mod r
   size_t universalHash(const std::string& key, const HashParams& params) const;

   // Constructor
   PerfectHashView(const std::vector<std::string>& keySet) {
      tableSize = keySet.size();
      lookupTable.resize(tableSize);
      lookupTableRaw.resize(tableSize);
   }

   // Build the perfect hash function
   static PerfectHashView* buildPerfectHash(const std::vector<std::string>& keySet);

   static PerfectHashView* build(FlexibleBuffer* lkbuffer, FlexibleBuffer* gvalues);

   // Universal hash function: h(x) = ((a*x + b) mod p) mod r
   size_t universalHash(const std::string& key, const HashParams& params, bool trim) const {
      size_t hash = 0;
      const uint32_t prime = 0x7FFFFFFF; // 2^31 - 1
      
      for (char c : key) {
         hash = (hash * params.a + static_cast<uint8_t>(c)) & prime;
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

      lingodb::runtime::VarLen32 key;
      std::memcpy(&key, keyPtr, sizeof(key));
      size_t h1 = universalHash(key, auxHashParams[0], false);
      size_t h2 = universalHash(key, auxHashParams[1], true);

      return (h1 + g[h2]);
   }

   // size_t computeHash(lingodb::runtime::VarLen32 key) {
   //    // printf("~~~ computeHash %p\n", keyPtr);
   //    // return 0;
   //    size_t h1 = universalHash(key, auxHashParams[0], false);
   //    size_t h2 = universalHash(key, auxHashParams[1], true);

   //    return (h1 + g[h2]);
   // }

   // Check if a key exists in the original key set
   void* containHash(size_t h) {
      static int collideNum = 0;
      size_t idx = h % tableSize;
      // if (h == lookupTable[idx].hashvalue) {
      //    collideNum ++;
      //    printf("~~~** containHash %lu %lu %s %lu, %lu\n", h, idx, lookupTable[idx].key.str().c_str(), lookupTable[idx].hashvalue, collideNum);
      // }
      
      // 直接检查查找表中的值是否匹配
      // TODO DELETE contains. LOOKUP logics in MLIR
      return &lookupTable[idx];
   }

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
