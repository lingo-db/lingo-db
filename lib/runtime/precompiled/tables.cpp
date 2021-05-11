#include "runtime/execution_context.h"
#include <arrow/array.h>
#define FORCE_INLINE __attribute__((always_inline))

extern "C" {
FORCE_INLINE uint64_t buffer_get_int_64(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((uint64_t*) buffer)[offset + i];
}
FORCE_INLINE uint32_t buffer_get_int_32(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((uint32_t*) buffer)[offset + i];
}
FORCE_INLINE double buffer_get_float_64(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((double*) buffer)[offset + i];
}
FORCE_INLINE float buffer_get_float_32(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((float*) buffer)[offset + i];
}
FORCE_INLINE __int128 buffer_get_decimal(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((__int128*) buffer)[offset + i];
}
FORCE_INLINE uint8_t* buffer_get_binary(db_table_column_buffer offsetBuffer, db_table_column_buffer varLenBuffer, int64_t offset, int64_t i, int64_t* len) {
   uint32_t* offsets = (uint32_t*) offsetBuffer;
   i += offset;
   const uint32_t pos = offsets[i];
   *len = offsets[i + 1] - pos;
   return varLenBuffer + pos;
}
FORCE_INLINE bool table_column_is_null(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   auto null_bitmap_data_=buffer;
   return null_bitmap_data_ != NULLPTR &&
          !arrow::BitUtil::GetBit(null_bitmap_data_, i + offset);
}
FORCE_INLINE bool buffer_get_bool(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return arrow::BitUtil::GetBit(buffer, i + offset);
}

}