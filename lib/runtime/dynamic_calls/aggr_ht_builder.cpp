#include "runtime/helpers.h"
#include "string.h"
#include <iostream>
#include <unordered_map>
#include <vector>
using i8 = uint8_t;
using i64 = uint64_t;
typedef bool (*compareFn)(i8*, i8*, i64, i8*, i8*, i64);
typedef void (*updateFn)(i8*, i8*, i64, i8*, i8*, i64);
struct AggrHashtableBuilder {
   std::string test = "abc";
   std::unordered_multimap<uint64_t, size_t> hashtable;
   std::vector<std::byte> keys;
   std::vector<std::byte> aggrs;
   size_t entries = 0;
   size_t keySize;
   size_t valSize;
   size_t aggrSize;
   compareFn compareFunction;
   updateFn updateFunction;
   std::vector<std::byte> initAggr;
   runtime::VarLenBuffer varLenBuffer;
   AggrHashtableBuilder(size_t keySize, size_t valSize, size_t aggrSize, compareFn compareFunction, updateFn updateFunction, std::byte* initAggr) : keySize(keySize), valSize(valSize), aggrSize(aggrSize), compareFunction(compareFunction), updateFunction(updateFunction) {
      this->initAggr.insert(this->initAggr.end(), initAggr, initAggr + aggrSize);
   }
   bool matches(std::byte* key1, std::byte* key2) {
      return compareFunction(nullptr, (uint8_t*) key1, 0, nullptr, (uint8_t*) key2, 0);
   }
   void update(std::byte* aggr, std::byte* val) {
      updateFunction(nullptr, (uint8_t*) aggr, 0, nullptr, (uint8_t*) val, 0);
   }

   void removeUneccesaryVarLen() {
   }
   std::string string_to_hex(const std::string& input) {
      static const char hex_digits[] = "0123456789ABCDEF";

      std::string output;
      output.reserve(input.length() * 2);
      for (unsigned char c : input) {
         output.push_back(hex_digits[c >> 4]);
         output.push_back(hex_digits[c & 15]);
      }
      return output;
   }

   std::string hex(std::byte* bytes, size_t len) {
      return string_to_hex(std::string((char*) bytes, len));
   }
   void insert(uint64_t hash, std::byte* key, std::byte* val) {
      auto range = hashtable.equal_range(hash);
      for (auto i = range.first; i != range.second; ++i) {
         auto [hash, offset] = *i;
         std::byte* entryKey = &keys[offset * keySize];
         std::byte* entryAggr = &aggrs[offset * aggrSize];
         if (matches(key, entryKey)) {
            update(entryAggr, val);
            removeUneccesaryVarLen();
            return;
         }
      }
      //no match found!
      keys.insert(keys.end(), key, key + keySize);
      size_t x = aggrs.size();
      aggrs.insert(aggrs.end(), initAggr.begin(), initAggr.end());
      update(&aggrs[x], val);
      hashtable.insert({hash, entries++});
   }

   runtime::Pair<runtime::ByteRange, runtime::ByteRange> scan() {
      return runtime::Pair(runtime::ByteRange((uint8_t*) &keys[0], keys.size()), runtime::ByteRange((uint8_t*) &aggrs[0], aggrs.size()));
   }
};

EXPORT runtime::Pointer<AggrHashtableBuilder> _mlir_ciface_aggr_ht_builder_create(uint64_t keySize, uint64_t valSize, uint64_t aggrSize, compareFn compareFunction, updateFn updateFunction, runtime::Pointer<std::byte>* bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto *ptr= new AggrHashtableBuilder(keySize, valSize, aggrSize, compareFunction, updateFunction, (*bytes).get());
   return ptr;
}
EXPORT runtime::ByteRange _mlir_ciface_aggr_ht_builder_add_var_len(runtime::Pointer<AggrHashtableBuilder>* builder, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->varLenBuffer.persist(*data);
}
EXPORT runtime::Pair<bool, runtime::ByteRange> _mlir_ciface_aggr_ht_builder_add_nullable_var_len(runtime::Pointer<AggrHashtableBuilder>* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::ByteRange(nullptr, 0)};
   }
   return {false, (*builder)->varLenBuffer.persist(*data)};
}

EXPORT void _mlir_ciface_aggr_ht_builder_merge(runtime::Pointer<AggrHashtableBuilder>* builder, uint64_t hash, runtime::Pointer<std::byte>* key, runtime::Pointer<std::byte>* val) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (*builder)->insert(hash, (*key).get(), (*val).get());
}
EXPORT runtime::Pointer<AggrHashtableBuilder> _mlir_ciface_aggr_ht_builder_build(runtime::Pointer<AggrHashtableBuilder>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return *builder;
}
EXPORT runtime::Pair<runtime::ByteRange, runtime::ByteRange> _mlir_ciface_aggr_ht_scan(runtime::Pointer<AggrHashtableBuilder>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->scan();
}