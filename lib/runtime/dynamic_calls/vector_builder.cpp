#include "runtime/helpers.h"
#include "string.h"
#include <iostream>
#include <vector>



EXPORT runtime::Pointer<runtime::Vector> _mlir_ciface_vector_builder_create() { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new runtime::Vector;
}
EXPORT runtime::ByteRange _mlir_ciface_vector_builder_add_var_len(runtime::Pointer<runtime::Vector>* builder, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->varLenBuffer.persist(*data);
}
EXPORT runtime::Pair<bool, runtime::ByteRange> _mlir_ciface_vector_builder_add_nullable_var_len(runtime::Pointer<runtime::Vector>* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::ByteRange(nullptr,0)};
   }
   return {false, (*builder)->varLenBuffer.persist(*data)};
}

EXPORT runtime::Pointer<uint8_t> _mlir_ciface_vector_builder_merge(runtime::Pointer<runtime::Vector>* builder, size_t bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& values = (*builder)->values;
   size_t sizeBefore = values.size();
   values.resize(sizeBefore + bytes);
   auto* ptr = &values[sizeBefore];

   return ptr;
}
EXPORT runtime::Pointer<runtime::Vector> _mlir_ciface_vector_builder_build(runtime::Pointer<runtime::Vector>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder).get();
}
EXPORT runtime::ByteRange _mlir_ciface_vector_get_values(runtime::Pointer<runtime::Vector>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& values= (*builder).get()->values;
   return runtime::ByteRange((uint8_t*) &values[0],values.size());
}