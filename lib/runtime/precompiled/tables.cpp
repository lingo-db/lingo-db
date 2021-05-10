#include "runtime/execution_context.h"
#include <arrow/array.h>
#define FORCE_INLINE __attribute__((always_inline))

extern "C" {
FORCE_INLINE uint64_t table_column_get_int_64(db_table_column_buffer buffer, int64_t offset, int64_t i) {
return ((uint64_t*) ((arrow::Buffer*) buffer)->data())[offset + i];
}

}