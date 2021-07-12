#include "runtime/helpers.h"
#include "runtime/aggr_hashtable.h"

using namespace runtime;

EXPORT AggrHashtableBuilder* _mlir_ciface_aggr_ht_builder_create(uint64_t keySize, uint64_t valSize, uint64_t aggrSize, uint64_t combinedSize, compareFn compareFunction, updateFn updateFunction, uint8_t* bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto* ptr = new AggrHashtableBuilder(keySize, valSize, aggrSize, combinedSize, compareFunction, updateFunction, bytes);
   return ptr;
}
EXPORT __attribute__((always_inline)) runtime::Bytes _mlir_ciface_aggr_ht_builder_add_var_len(AggrHashtableBuilder* builder, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->currVarLen += (data).getSize();
   return (builder)->varLenBuffer.persist(data);
}
EXPORT __attribute__((always_inline)) runtime::Pair<bool, runtime::Bytes> _mlir_ciface_aggr_ht_builder_add_nullable_var_len(AggrHashtableBuilder* builder, bool null, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::Bytes(nullptr, 0)};
   }
   (builder)->currVarLen += (data).getSize();
   return {false, (builder)->varLenBuffer.persist(data)};
}
EXPORT __attribute__((always_inline)) runtime::Pair<bool, uint8_t*> _mlir_ciface_aggr_ht_builder_fast_lookup(AggrHashtableBuilder* builder, uint64_t hash) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (builder)->lookup(hash);
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_aggr_ht_builder_merge(AggrHashtableBuilder* builder, uint64_t hash, uint8_t* key, uint8_t* val) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->insert(hash, key, val);
}
EXPORT __attribute__((always_inline)) AggrHashtableBuilder* _mlir_ciface_aggr_ht_builder_build(AggrHashtableBuilder* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return builder;
}
using iterator_t = runtime::ObjectBuffer<SimpleHashTable::Entry>::RangeIterator;
EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_aggr_ht_iterator_init(AggrHashtableBuilder* hashmap) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto x = (hashmap)->getHashTable().getBuffer().beginPtr();
   return x;
}
EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_aggr_ht_iterator_next(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (iterator)->operator++();
   return iterator;
}
EXPORT __attribute__((always_inline)) uint8_t* _mlir_ciface_aggr_ht_iterator_curr(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   uint8_t* ptr = (*iterator)->data;
   return ptr;
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_aggr_ht_iterator_valid(iterator_t* iterator) {
   return iterator->valid();
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_aggr_ht_iterator_free(iterator_t* iterator) {
   delete iterator;
}