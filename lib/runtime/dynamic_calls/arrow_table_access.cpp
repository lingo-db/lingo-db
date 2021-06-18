#include "runtime/helpers.h"
#include <arrow/array.h>
#include <arrow/table.h>

EXPORT uint64_t _mlir_ciface_get_column_id(runtime::Pointer<std::shared_ptr<arrow::Table>>* table, runtime::String* columnName) {
   auto column_names = (*table).ref()->ColumnNames();
   size_t column_id = 0;
   for (auto column : column_names) {
      if (column == columnName->str()) {
         return column_id;
      }
      column_id++;
   }
   throw std::runtime_error("column not found: "+columnName->str());
}
struct tableChunkIteratorStruct {
   arrow::TableBatchReader reader;
   std::shared_ptr<arrow::RecordBatch> curr_chunk;
   tableChunkIteratorStruct(arrow::Table& table) : reader(table), curr_chunk() {}
};

EXPORT runtime::Pointer<tableChunkIteratorStruct> _mlir_ciface_table_chunk_iterator_init(runtime::Pointer<std::shared_ptr<arrow::Table>>* table) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto* tableChunkIterator = new tableChunkIteratorStruct(*(*table).ref());
   if (tableChunkIterator->reader.ReadNext(&tableChunkIterator->curr_chunk) != arrow::Status::OK()) {
      tableChunkIterator->curr_chunk.reset();
   }
   return tableChunkIterator;
}
EXPORT runtime::Pointer<tableChunkIteratorStruct> _mlir_ciface_table_chunk_iterator_next(runtime::Pointer<tableChunkIteratorStruct>* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if ((*iterator)->reader.ReadNext(&(*iterator)->curr_chunk) != arrow::Status::OK()) {
      (*iterator)->curr_chunk.reset();
   }
   return *iterator;
}
EXPORT runtime::Pointer<arrow::RecordBatch> _mlir_ciface_table_chunk_iterator_curr(runtime::Pointer<tableChunkIteratorStruct>* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*iterator)->curr_chunk.get();
}

EXPORT bool _mlir_ciface_table_chunk_iterator_valid(runtime::Pointer<tableChunkIteratorStruct>* iterator) {
   auto valid = (*iterator)->curr_chunk.operator bool();
   return valid;
}
EXPORT void _mlir_ciface_table_chunk_iterator_free(runtime::Pointer<tableChunkIteratorStruct>* iterator) {
   return delete iterator->get();
}
EXPORT uint64_t _mlir_ciface_table_chunk_num_rows(runtime::Pointer<arrow::RecordBatch>* tableChunk) {
   return (*tableChunk)->num_rows();
}

EXPORT runtime::ByteRange _mlir_ciface_table_chunk_get_column_buffer(runtime::Pointer<arrow::RecordBatch>* tableChunk, uint64_t columnId, uint64_t bufferId) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   static uint8_t alternative = 0b11111111;
   auto* buffer = (*tableChunk)->column_data(columnId)->buffers[bufferId].get();
   if (buffer) {
      uint8_t* data = (uint8_t*) buffer->address();
      size_t len = (*tableChunk)->column_data(columnId)->buffers[bufferId].get()->size();

      return {data, len};
   } else {
      return {&alternative, 0}; //always return valid pointer to at least one byte filled with ones
   }
}
EXPORT uint64_t _mlir_ciface_table_chunk_get_column_offset(runtime::Pointer<arrow::RecordBatch>* tableChunk, uint64_t columnId) {
   return (*tableChunk)->column_data(columnId)->offset;
}