#include "runtime/execution_context.h"
#include <arrow/array.h>
#define FORCE_INLINE __attribute__((always_inline))

extern "C" {
FORCE_INLINE uint64_t _mlir_ciface_buffer_get_int_64(runtime::Pointer<uint64_t>* buffer, int64_t offset, int64_t i) {
   return (*buffer)[offset + i];
}
FORCE_INLINE uint32_t _mlir_ciface_buffer_get_int_32(runtime::Pointer<uint32_t>* buffer, int64_t offset, int64_t i) {
   return (*buffer)[offset + i];
}
FORCE_INLINE double _mlir_ciface_buffer_get_float_64(runtime::Pointer<double>* buffer, int64_t offset, int64_t i) {
   return (*buffer)[offset + i];
}
FORCE_INLINE float _mlir_ciface_buffer_get_float_32(runtime::Pointer<float>* buffer, int64_t offset, int64_t i) {
   return (*buffer)[offset + i];
}
FORCE_INLINE __int128 _mlir_ciface_buffer_get_decimal(runtime::Pointer<__int128>* buffer, int64_t offset, int64_t i) {
   return (*buffer)[offset + i];
}
FORCE_INLINE runtime::Pointer<uint8_t> _mlir_ciface_buffer_get_binary(runtime::Pointer<uint32_t>* offsetBuffer, runtime::Pointer<uint8_t>* varLenBuffer, int64_t offset, int64_t i, runtime::Pointer<int64_t>* len) {
   i += offset;
   const uint32_t pos = (*offsetBuffer)[i];
   **len = (*offsetBuffer)[i + 1] - pos;
   return varLenBuffer->get() + pos;
}
FORCE_INLINE bool _mlir_ciface_table_column_is_null(runtime::Pointer<uint8_t>* buffer, int64_t offset, int64_t i) {
   auto null_bitmap_data_ = buffer->get();
   return null_bitmap_data_ != NULLPTR &&
      !arrow::BitUtil::GetBit(null_bitmap_data_, i + offset);
}
FORCE_INLINE bool _mlir_ciface_buffer_get_bool(runtime::Pointer<uint8_t>* buffer, int64_t offset, int64_t i) {
   return arrow::BitUtil::GetBit(buffer->get(), i + offset);
}
}