#ifndef RUNTIME_RECORDBATCHINFO_H
#define RUNTIME_RECORDBATCHINFO_H
#include <cstddef>
#include <cstdint>
#pragma once
namespace arrow {
class RecordBatch;
} // end namespace arrow
namespace runtime {
struct ColumnInfo {
   // Offset of tuple/record batch
   size_t offset;
   // Needed in case arrow omits valid buffer (always true -> validMultiplier=0)
   size_t validMultiplier;
   // Pointer to compact representation of null values
   uint8_t* validBuffer;
   // Pointer to fixed size data/offsets for variable sized data
   uint8_t* dataBuffer;
   // Pointer to  variable size data
   uint8_t* varLenBuffer;
};
struct RecordBatchInfo {
   size_t numRows;
   ColumnInfo columnInfo[];

   // Access buffers for record batch and handle case where valid buffer is omitted
   static uint8_t* getBuffer(arrow::RecordBatch* batch, size_t columnId, size_t bufferId);
};

} // end namespace runtime
#endif //RUNTIME_RECORDBATCHINFO_H