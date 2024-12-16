#include "lingodb/runtime/RecordBatchInfo.h"
#include <arrow/array.h>
#include <arrow/record_batch.h>

uint8_t* lingodb::runtime::RecordBatchInfo::getBuffer(arrow::RecordBatch* batch, size_t columnId, size_t bufferId)  {
   static uint8_t alternative = 0b11111111;
   if (batch->column_data(columnId)->buffers.size() > bufferId && batch->column_data(columnId)->buffers[bufferId]) {
      auto* buffer = batch->column_data(columnId)->buffers[bufferId].get();
      return (uint8_t*) buffer->address();
   } else {
      return &alternative; //always return valid pointer to at least one byte filled with ones
   }
}