#include "runtime/helpers.h"
#include "string.h"
#include <iostream>
#include <vector>

struct VectorBuilder {

   std::vector<std::byte> values;
   std::vector<std::byte> rawValues;
   runtime::Pair<runtime::ByteRange, runtime::ByteRange> build() {
      auto* valuesArr = new uint8_t[values.size()];
      auto* rawValuesArr = new uint8_t[rawValues.size()];
      memcpy(valuesArr, &(values[0]), values.size());
      memcpy(rawValuesArr, &(rawValues[0]), rawValues.size());
      return runtime::Pair(runtime::ByteRange(valuesArr, values.size()), runtime::ByteRange(rawValuesArr, rawValues.size()));
   }
};

EXPORT runtime::Pointer<VectorBuilder> _mlir_ciface_vector_builder_create() { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new VectorBuilder;
}
EXPORT void _mlir_ciface_vector_builder_add_var_len(runtime::Pair<uint64_t, uint64_t>* result, runtime::Pointer<VectorBuilder>* builder, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& rawValues = (*builder)->rawValues;
   size_t sizeBefore = rawValues.size();
   rawValues.resize(sizeBefore + (*data).getSize());
   memcpy(&rawValues[sizeBefore], (*data).getPtr(), (*data).getSize());
   *result = {sizeBefore, (*data).getSize()};
}
EXPORT runtime::Triple<bool, uint64_t, uint64_t> _mlir_ciface_vector_builder_add_nullable_var_len(runtime::Pointer<VectorBuilder>* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, 0, 0};
   }
   auto& rawValues = (*builder)->rawValues;
   size_t sizeBefore = rawValues.size();
   rawValues.resize(sizeBefore + (*data).getSize());
   memcpy(&rawValues[sizeBefore], (*data).getPtr(), (*data).getSize());
   return {false, sizeBefore, (*data).getSize()};
}

EXPORT runtime::Pointer<std::byte> _mlir_ciface_vector_builder_merge(runtime::Pointer<VectorBuilder>* builder, size_t bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& values = (*builder)->values;
   size_t sizeBefore = values.size();
   values.resize(sizeBefore + bytes);
   std::byte* ptr = &values[sizeBefore];

   return ptr;
}
EXPORT runtime::Pair<runtime::ByteRange, runtime::ByteRange> _mlir_ciface_vector_builder_build(runtime::Pointer<VectorBuilder>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->build();
}