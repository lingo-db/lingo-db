#include "runtime/helpers.h"
#include "string.h"
#include <iostream>
#include <unordered_map>
#include <vector>
using i8 = uint8_t;
using i64 = uint64_t;
typedef bool (*compareFn)(i8*, i8*, i64, i64, i64, i8*, i8*, i64, i8*, i8*, i64);
typedef void (*updateFn)(i8*, i8*, i64, i64, i64, i8*, i8*, i64, i8*, i8*, i64);
struct AggrHashtableBuilder {
   std::unordered_multimap<uint64_t, size_t> hashtable;
   std::vector<std::byte> keys;
   std::vector<std::byte> values;
   std::vector<std::byte> rawValues;
   size_t entries = 0;
   size_t keySize;
   size_t valSize;
   compareFn compareFunction;
   updateFn updateFunction;
   AggrHashtableBuilder(size_t keySize, size_t valSize, compareFn compareFunction, updateFn updateFunction) : keySize(keySize), valSize(valSize), compareFunction(compareFunction), updateFunction(updateFunction) {
   }
   bool matches(std::byte* key1, std::byte* key2) {
      std::cout<<"matches ("<<hex(key1,keySize)<<", "<<hex(key2,keySize)<<std::endl;

      return compareFunction(nullptr, (uint8_t*) &rawValues[0], 0, rawValues.size(), 0, nullptr, (uint8_t*) key1, 0, nullptr, (uint8_t*) key2, 0);
   }
   void update(std::byte* val1, std::byte* val2) {
      std::cout<<"update ("<<hex(val1,valSize)<<", "<<hex(val2,valSize)<<std::endl;
      updateFunction(nullptr, (uint8_t*) &rawValues[0], 0, rawValues.size(), 0, nullptr, (uint8_t*) val1, 0, nullptr, (uint8_t*) val2, 0);
   }
   void removeUneccesaryVarLen() {
   }
   std::string string_to_hex(const std::string& input)
   {
      static const char hex_digits[] = "0123456789ABCDEF";

      std::string output;
      output.reserve(input.length() * 2);
      for (unsigned char c : input)
      {
         output.push_back(hex_digits[c >> 4]);
         output.push_back(hex_digits[c & 15]);
      }
      return output;
   }


   std::string hex(std::byte* bytes, size_t len){
      return string_to_hex(std::string((char*)bytes,len));
   }
   void insert(uint64_t hash, std::byte* key, std::byte* val) {
      std::cout<<"insert ("<<hash<<", "<<hex(key,keySize)<<", "<<hex(val,valSize)<<std::endl;
      auto range = hashtable.equal_range(hash);
      for (auto i = range.first; i != range.second; ++i) {
         auto [hash, offset] = *i;
         std::byte* entryKey = &keys[offset * keySize];
         std::byte* entryVal = &values[offset * valSize];
         if (matches(key, entryKey)) {
            update(entryVal, val);
            removeUneccesaryVarLen();
            return;
         }
      }
      //no match found!
      keys.insert(keys.end(), key, key + keySize);
      values.insert(values.end(), val, val + valSize);
      hashtable.insert({hash, entries++});
   }

   runtime::Triple<runtime::ByteRange, runtime::ByteRange, runtime::ByteRange> scan() {
      return runtime::Triple(runtime::ByteRange((uint8_t *)&keys[0], keys.size()), runtime::ByteRange((uint8_t *)&values[0], values.size()), runtime::ByteRange((uint8_t *)&rawValues[0], rawValues.size()));
   }
};

EXPORT runtime::Pointer<AggrHashtableBuilder> _mlir_ciface_aggr_ht_builder_create(uint64_t keySize, uint64_t valSize,compareFn compareFunction,updateFn updateFunction ) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new AggrHashtableBuilder(keySize,valSize,compareFunction,updateFunction);
}
EXPORT void _mlir_ciface_aggr_ht_builder_add_var_len(runtime::Pair<uint64_t, uint64_t>* result, runtime::Pointer<AggrHashtableBuilder>* builder, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& rawValues = (*builder)->rawValues;
   size_t sizeBefore = rawValues.size();
   rawValues.resize(sizeBefore + (*data).getSize());
   memcpy(&rawValues[sizeBefore], (*data).getPtr(), (*data).getSize());
   *result = {sizeBefore, (*data).getSize()};
}
EXPORT runtime::Triple<bool, uint64_t, uint64_t> _mlir_ciface_aggr_ht_builder_add_nullable_var_len(runtime::Pointer<AggrHashtableBuilder>* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, 0, 0};
   }
   auto& rawValues = (*builder)->rawValues;
   size_t sizeBefore = rawValues.size();
   rawValues.resize(sizeBefore + (*data).getSize());
   memcpy(&rawValues[sizeBefore], (*data).getPtr(), (*data).getSize());
   return {false, sizeBefore, (*data).getSize()};
}

EXPORT void _mlir_ciface_aggr_ht_builder_merge(runtime::Pointer<AggrHashtableBuilder>* builder, uint64_t hash,runtime::Pointer<std::byte>* key,runtime::Pointer<std::byte>* val) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (*builder)->insert(hash,(*key).get(),(*val).get());
}
EXPORT runtime::Pointer<AggrHashtableBuilder> _mlir_ciface_aggr_ht_builder_build(runtime::Pointer<AggrHashtableBuilder>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return *builder;
}
EXPORT runtime::Triple<runtime::ByteRange, runtime::ByteRange,runtime::ByteRange> _mlir_ciface_aggr_ht_scan(runtime::Pointer<AggrHashtableBuilder>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->scan();
}