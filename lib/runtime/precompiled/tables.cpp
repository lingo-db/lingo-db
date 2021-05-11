#include "runtime/execution_context.h"
#include <arrow/array.h>
#define FORCE_INLINE __attribute__((always_inline))

extern "C" {
FORCE_INLINE uint64_t table_column_get_int_64(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((uint64_t*) ((arrow::Buffer*) buffer)->data())[offset + i];
}
FORCE_INLINE uint32_t table_column_get_int_32(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((uint32_t*) ((arrow::Buffer*) buffer)->data())[offset + i];
}
FORCE_INLINE double table_column_get_float_64(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((double*) ((arrow::Buffer*) buffer)->data())[offset + i];
}
FORCE_INLINE float table_column_get_float_32(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((float*) ((arrow::Buffer*) buffer)->data())[offset + i];
}
FORCE_INLINE __int128 table_column_get_decimal(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return ((__int128*) ((arrow::Buffer*) buffer)->data())[offset + i];
}
FORCE_INLINE uint8_t* table_column_get_binary(db_table_column_buffer offsetBuffer, db_table_column_buffer varLenBuffer, int64_t offset, int64_t i, int64_t* len) {
   uint32_t* offsets = (uint32_t*) ((arrow::Buffer*) offsetBuffer)->data();
   uint8_t* data = (uint8_t*) ((arrow::Buffer*) varLenBuffer)->data();
   i += offset;
   const uint32_t pos = offsets[i];
   *len = offsets[i + 1] - pos;
   return data + pos;
}
FORCE_INLINE bool table_column_is_null(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   auto null_bitmap_data_=((arrow::Buffer*) buffer)->data();
   return null_bitmap_data_ != NULLPTR &&
          !arrow::BitUtil::GetBit(null_bitmap_data_, i + offset);
}
FORCE_INLINE bool table_column_get_bool(db_table_column_buffer buffer, int64_t offset, int64_t i) {
   return arrow::BitUtil::GetBit(((arrow::Buffer*) buffer)->data(), i + offset);
}

}