#include "runtime/execution_context.h"
#include <arrow/array.h>
#define FORCE_INLINE __attribute__((always_inline))

extern "C" {
FORCE_INLINE uint64_t buffer_get_int_64(uint64_t* buffer, int64_t offset, int64_t i) {
   return buffer[offset + i];
}
FORCE_INLINE uint32_t buffer_get_int_32(uint32_t* buffer, int64_t offset, int64_t i) {
   return buffer[offset + i];
}
FORCE_INLINE double buffer_get_float_64(double* buffer, int64_t offset, int64_t i) {
   return buffer[offset + i];
}
FORCE_INLINE float buffer_get_float_32(float* buffer, int64_t offset, int64_t i) {
   return ((float*) buffer)[offset + i];
}
FORCE_INLINE __int128 buffer_get_decimal(__int128* buffer, int64_t offset, int64_t i) {
   return ((__int128*) buffer)[offset + i];
}
FORCE_INLINE uint8_t* buffer_get_binary(uint32_t* offsetBuffer, uint8_t* varLenBuffer, int64_t offset, int64_t i, int64_t* len) {
   i += offset;
   const uint32_t pos = offsetBuffer[i];
   *len = offsetBuffer[i + 1] - pos;
   return varLenBuffer + pos;
}
FORCE_INLINE bool table_column_is_null(uint8_t* buffer, int64_t offset, int64_t i) {
   auto null_bitmap_data_=buffer;
   return null_bitmap_data_ != NULLPTR &&
          !arrow::BitUtil::GetBit(null_bitmap_data_, i + offset);
}
FORCE_INLINE bool buffer_get_bool(uint8_t* buffer, int64_t offset, int64_t i) {
   return arrow::BitUtil::GetBit(buffer, i + offset);
}

}