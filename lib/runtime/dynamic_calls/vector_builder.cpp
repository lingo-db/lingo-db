#include "runtime/helpers.h"
#include "string.h"
#include <vector>
runtime::Vector* cachedRuntimeVector = nullptr; //hack for now, todo: fix in the future

EXPORT runtime::Vector* _mlir_ciface_vector_builder_create() { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (cachedRuntimeVector) {
      auto *res = cachedRuntimeVector;
      cachedRuntimeVector = nullptr;
      return res;
   }
   return new runtime::Vector;
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
   auto& values = (builder)->values;
   return runtime::Bytes((uint8_t*) &values[0], values.size());
}
EXPORT void _mlir_ciface_vector_free(runtime::Vector* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if(!cachedRuntimeVector){
      cachedRuntimeVector=builder;
      cachedRuntimeVector->values.clear();
      cachedRuntimeVector->varLenBuffer.clear();
   }else {
      delete builder;
   }
}