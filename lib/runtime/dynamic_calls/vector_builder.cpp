#include "runtime/helpers.h"
#include "string.h"
#include <iostream>
#include <vector>

EXPORT runtime::Vector* _mlir_ciface_vector_builder_create() { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new runtime::Vector;
}
EXPORT runtime::Bytes _mlir_ciface_vector_builder_add_var_len(runtime::Vector* builder, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (builder)->varLenBuffer.persist(data);
}
EXPORT runtime::Pair<bool, runtime::Bytes> _mlir_ciface_vector_builder_add_nullable_var_len(runtime::Vector* builder, bool null, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::Bytes(nullptr,0)};
   }
   return {false, (builder)->varLenBuffer.persist(data)};
}

EXPORT uint8_t* _mlir_ciface_vector_builder_merge(runtime::Vector* builder, size_t bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& values = (builder)->values;
   size_t sizeBefore = values.size();
   values.resize(sizeBefore + bytes);
   auto* ptr = &values[sizeBefore];

   return ptr;
}
EXPORT runtime::Vector* _mlir_ciface_vector_builder_build(runtime::Vector* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return builder;
}
EXPORT runtime::Bytes _mlir_ciface_vector_get_values(runtime::Vector* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& values= (builder)->values;
   return runtime::Bytes((uint8_t*) &values[0],values.size());
}