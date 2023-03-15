#ifndef RUNTIME_RECORDBATCHINFO_H
#define RUNTIME_RECORDBATCHINFO_H
#include <arrow/array.h>
#include <arrow/record_batch.h>
#pragma once
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
      static uint8_t* getBuffer(arrow::RecordBatch* batch, size_t columnId, size_t bufferId) {
         static uint8_t alternative = 0b11111111;
         if (batch->column_data(columnId)->buffers.size() > bufferId && batch->column_data(columnId)->buffers[bufferId]) {
            auto* buffer = batch->column_data(columnId)->buffers[bufferId].get();
            return (uint8_t*) buffer->address();
         } else {
            return &alternative; //always return valid pointer to at least one byte filled with ones
         }
      }
   };

} // end namespace runtime
#endif //RUNTIME_RECORDBATCHINFO_H