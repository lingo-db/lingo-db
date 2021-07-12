#include "runtime/helpers.h"

EXPORT runtime::Bytes _mlir_ciface_vector_builder_add_var_len(runtime::Vector* builder, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (builder)->varLenBuffer.persist(data);
}
EXPORT runtime::Bytes _mlir_ciface_vector_builder_add_nullable_var_len(runtime::Vector* builder, bool null, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return runtime::Bytes(nullptr, 0);
   }
   return (builder)->varLenBuffer.persist(data);
}