#include "runtime/helpers.h"
#include "runtime/join_hashtables.h"
#include "string.h"
using runtime::LazyMultiMap;

EXPORT LazyMultiMap* _mlir_ciface_join_ht_builder_create(size_t dataSize) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new LazyMultiMap(dataSize);
}
EXPORT __attribute__((always_inline)) runtime::Bytes _mlir_ciface_join_ht_builder_add_var_len(LazyMultiMap* builder, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return builder->varLenBuffer.persist(data);
}
EXPORT __attribute__((always_inline)) runtime::Pair<bool, runtime::Bytes> _mlir_ciface_join_ht_builder_add_nullable_var_len(LazyMultiMap* builder, bool null, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::Bytes(nullptr, 0)};
   }
   return {false, builder->varLenBuffer.persist(data)};
}

EXPORT __attribute__((always_inline)) LazyMultiMap::Entry* _mlir_ciface_join_ht_builder_merge(LazyMultiMap* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   builder->entries++;
   return builder->entryBuffer.alloc();
}
EXPORT LazyMultiMap* _mlir_ciface_join_ht_builder_build(LazyMultiMap* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   builder->finalize();
   return builder;
}

EXPORT __attribute__((always_inline)) LazyMultiMap::Entry* _mlir_ciface_join_ht_iterator_init(LazyMultiMap* hashmap, size_t hash) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (hashmap)->getIt(hash);
}
EXPORT __attribute__((always_inline)) LazyMultiMap::Entry* _mlir_ciface_join_ht_iterator_next(LazyMultiMap::Entry* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (iterator)->next;
}
EXPORT __attribute__((always_inline)) uint8_t* _mlir_ciface_join_ht_iterator_curr(LazyMultiMap::Entry* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (iterator)->data;
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_join_ht_iterator_valid(LazyMultiMap::Entry* iterator) {
   return iterator != nullptr;
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_join_ht_iterator_free(LazyMultiMap::Entry* iterator) {
}
