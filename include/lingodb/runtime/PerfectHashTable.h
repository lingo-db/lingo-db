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

   // Hash function to map a key to its perfect hash index
   size_t hash(const std::string& key) const;

   // size_t computeHash(lingodb::runtime::VarLen32 key) {
   //    return hash(key);
   // }
   size_t computeHash(uint8_t* keyPtr) {
      // printf("~~~ computeHash %p\n", keyPtr);
      // return 0;
      if (keyPtr == nullptr) {
         printf("~~~!! computeHash %p\n", keyPtr);
      }
      lingodb::runtime::VarLen32 key;
      std::memcpy(&key, keyPtr, sizeof(key));
      auto h = hash(key);
      // printf("~~~ computeHash %s %lu\n", key.str().c_str(), h);
      return h;

      // lingodb::runtime::VarLen32 key;
      // std::memcpy(&key, keyPtr, sizeof(key));
      // size_t hash = 0;
      // const uint32_t prime = 0x7FFFFFFF; // 2^31 - 1

      // hash = (hash * auxHashParams[0].a + static_cast<uint8_t>(key.getPtr()[key.getLen()-1])) & prime;
      // hash = (hash * auxHashParams[0].b) & prime;
      // size_t h1 = hash % this->r;

      // hash = 0;
      // hash = (hash * auxHashParams[1].a + static_cast<uint8_t>(key.getPtr()[key.getLen()-1])) & prime;
      // hash = (hash * auxHashParams[1].b) & prime;
      // size_t h2 = hash % this->r;

      // return (h1 + g[h2]);
   }

   // Check if a key exists in the original key set
   void* containHash(size_t h) {
      size_t idx = h % tableSize;
      // printf("~~~** containHash %lu %lu %s %lu\n", h, idx, lookupTable[idx].key.str().c_str(), lookupTable[idx].hashvalue);
      
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
