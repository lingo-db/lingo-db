#include "runtime/helpers.h"
#include "runtime/join_hashtables.h"
#include "string.h"
using namespace runtime;

EXPORT MarkableLazyMultiMap* _mlir_ciface_mjoin_ht_builder_create(size_t dataSize) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new MarkableLazyMultiMap(dataSize);
}
EXPORT __attribute__((always_inline)) runtime::Bytes _mlir_ciface_mjoin_ht_builder_add_var_len(MarkableLazyMultiMap* builder, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (builder)->varLenBuffer.persist(data);
}
EXPORT __attribute__((always_inline)) runtime::Bytes _mlir_ciface_mjoin_ht_builder_add_nullable_var_len(MarkableLazyMultiMap* builder, bool null, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return runtime::Bytes(nullptr, 0);
   }
   return (builder)->varLenBuffer.persist(data);
}

EXPORT __attribute__((always_inline)) uint8_t* _mlir_ciface_mjoin_ht_builder_merge(MarkableLazyMultiMap* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->entries++;
   return (uint8_t*) &(builder)->entryBuffer.alloc()->next;
}
EXPORT MarkableLazyMultiMap* _mlir_ciface_mjoin_ht_builder_build(MarkableLazyMultiMap* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->finalize();
   return builder;
}
//for lookups
EXPORT __attribute__((always_inline)) MarkableLazyMultiMap::Entry* _mlir_ciface_mjoin_ht_lookup_iterator_init(MarkableLazyMultiMap* hashmap, size_t hash) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (hashmap)->getIt(hash);
}
EXPORT __attribute__((always_inline)) MarkableLazyMultiMap::Entry* _mlir_ciface_mjoin_ht_lookup_iterator_next(MarkableLazyMultiMap::Entry* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return iterator->next;
}
EXPORT __attribute__((always_inline)) runtime::Pair<uint8_t*,uint8_t*> _mlir_ciface_mjoin_ht_lookup_iterator_curr(MarkableLazyMultiMap::Entry* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return {(iterator)->data,&(iterator)->mark};
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_mjoin_ht_lookup_iterator_valid(MarkableLazyMultiMap::Entry* iterator) {
   return iterator != nullptr;
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_mjoin_ht_lookup_iterator_free(MarkableLazyMultiMap::Entry* iterator) {
}
using iterator_t = runtime::ObjectBuffer<MarkableLazyMultiMap::Entry>::RangeIterator;

EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_mjoin_ht_iterator_init(MarkableLazyMultiMap* hashmap) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (hashmap)->getBuffer().beginPtr();
}
EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_mjoin_ht_iterator_next(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   iterator->operator++();
   return iterator;
}
EXPORT __attribute__((always_inline)) runtime::Pair<uint8_t*,uint8_t*> _mlir_ciface_mjoin_ht_iterator_curr(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return {(*iterator)->data,&(*iterator)->mark};
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_mjoin_ht_iterator_valid(iterator_t* iterator) {
   return iterator->valid();
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_mjoin_ht_iterator_free(iterator_t* iterator) {
   delete iterator;
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_mjoin_ht_free(MarkableLazyMultiMap* mjoinHt) {
   delete mjoinHt;
}