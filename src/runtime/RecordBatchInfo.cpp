#include "lingodb/runtime/RecordBatchInfo.h"
#include <arrow/array.h>
#include <arrow/record_batch.h>
namespace {
uint8_t* getBufferOrAlternative(std::shared_ptr<arrow::ArrayData> arrayData, size_t bufferId) {
   static uint8_t alternative = 0b11111111;
   if (arrayData->buffers.size() > bufferId && arrayData->buffers[bufferId]) {
      auto* buffer = arrayData->buffers[bufferId].get();
      return (uint8_t*) buffer->address();
   } else {
      return &alternative; //always return valid pointer to at least one byte filled with ones
   }
}
} //end namespace

uint8_t* lingodb::runtime::RecordBatchInfo::getBuffer(arrow::RecordBatch* batch, size_t columnId, size_t bufferId) {
   static uint8_t alternative = 0b11111111;
   if (batch->column_data(columnId)->buffers.size() > bufferId && batch->column_data(columnId)->buffers[bufferId]) {
      auto* buffer = batch->column_data(columnId)->buffers[bufferId].get();
      return (uint8_t*) buffer->address();
   } else {
      return &alternative; //always return valid pointer to at least one byte filled with ones
   }
}

void lingodb::runtime::RecordBatchInfo::access(std::vector<size_t> colIds, lingodb::runtime::RecordBatchInfo* info, const std::shared_ptr<arrow::RecordBatch>& currChunk) {
   for (size_t i = 0; i < colIds.size(); i++) {
      auto colId = colIds[i];
      lingodb::runtime::ColumnInfo& colInfo = info->columnInfo[i];
      size_t off = currChunk->column_data(colId)->offset;
      colInfo.offset = off;
      colInfo.validMultiplier = currChunk->column_data(colId)->buffers[0] ? 1 : 0;
      colInfo.validBuffer = lingodb::runtime::RecordBatchInfo::getBuffer(currChunk.get(), colId, 0);
      colInfo.dataBuffer = lingodb::runtime::RecordBatchInfo::getBuffer(currChunk.get(), colId, 1);
      colInfo.varLenBuffer = lingodb::runtime::RecordBatchInfo::getBuffer(currChunk.get(), colId, 2);
      if (currChunk->column(colId)->type()->id() == arrow::Type::LIST) {
         auto childData = currChunk->column_data(colId)->child_data[0];
         colInfo.childInfo = new lingodb::runtime::ColumnInfo; //todo: fix
         colInfo.childInfo->offset = childData->offset;
         colInfo.childInfo->validMultiplier = childData->buffers[0] ? 1 : 0;
         colInfo.childInfo->validBuffer = getBufferOrAlternative(childData, 0);
         colInfo.childInfo->dataBuffer = getBufferOrAlternative(childData, 1);
         colInfo.childInfo->varLenBuffer = getBufferOrAlternative(childData, 2);
      } else {
         colInfo.childInfo = nullptr;
      }
   }
   info->numRows = currChunk->num_rows();
}