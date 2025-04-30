#ifndef LINGODB_RUNTIME_PERFECTHASHTABLE_H
#define LINGODB_RUNTIME_PERFECTHASHTABLE_H
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/StringRuntime.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"

namespace lingodb::runtime {
class GrowingBuffer;
struct HashParams {
   uint32_t a;
   uint32_t b;
};

// struct PerfectHashCombination {
//    // TODO GrowingBuffer for int8_t
//    std::vector<int8_t> g;                  // Displacement values
//    // TODO GrowingBuffer for VarLen32
//    std::vector<std::optional<std::string>> lookupTable;
//    std::array<HashParams, 2> auxHashParams;  // Parameters for auxiliary hash functions
//    size_t tableSize;                       // Size of the hash table (number of keys)
//    size_t r;                               // Range for intermediate hash values
// };

// TODO RENAME
struct LKEntry {
   uint8_t content[];
};
struct GEntry {
   size_t index;
   size_t value;
};

class PerfectHashView {
   // Hash function parameters
   // HashParams auxHashParams[2];  // Parameters for auxiliary hash functions
   // int8_t* g;                  // Displacement values
   // lingodb::runtime::VarLen32** lookupTable;
   // size_t tableSize;                       // Size of the hash table (number of keys)
   // size_t r;                               // Range for intermediate hash values
public:
   std::vector<LKEntry*> lookupTable;
   std::vector<size_t> g;                  // Displacement values
   HashParams auxHashParams[2];  // Parameters for auxiliary hash functions
   size_t tableSize;                       // Size of the hash table (number of keys)
   size_t r;
   
   std::vector<std::optional<std::string>> lookupTableRaw;

   // Universal hash function: h(x) = ((a*x + b) mod p) mod r
   size_t universalHash(const std::string& key, const HashParams& params) const;

   // Constructor
   PerfectHashView(const std::vector<std::string>& keySet) {
      tableSize = keySet.size();
      lookupTable.resize(tableSize);
      // For FCH, r is typically set to m² where m is number of keys
      r = std::max(size_t(16), tableSize * tableSize);
   }

   mlir::Value lingodb::runtime::PerfectHashView::convertToMLIRValue(PerfectHashView& instance, mlir::OpBuilder& builder);

   // Build the perfect hash function
   static PerfectHashView* buildPerfectHash(const std::vector<std::string>& keySet);

   static PerfectHashView* build(FlexibleBuffer* lkbuffer, FlexibleBuffer* gvalues, FlexibleBuffer* auxvalues);

   // Hash function to map a key to its perfect hash index
   size_t hash(const std::string& key) const;

   // Check if a key exists in the original key set
   bool contains(lingodb::runtime::VarLen32 key) const {
      size_t idx = hash(key);
      if (idx >= tableSize) return false;
      
      // 直接检查查找表中的值是否匹配
      // TODO DELETE contains. LOOKUP logics in MLIR
      return true;
      // return !lookupTable[idx]->empty() && lingodb::runtime::StringRuntime::compareEq(lookupTable[idx].value(), key);
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
