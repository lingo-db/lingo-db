#include "json.h"
#include "runtime/execution_context.h"
#include "runtime/helpers.h"
#include <iostream>
#include <arrow/array.h>
#include <arrow/table.h>
EXPORT uint64_t get_column_id(std::shared_ptr<arrow::Table>* table, std::string columnName) {
   auto column_names = (*table)->ColumnNames();
   size_t column_id = 0;
   for (auto column : column_names) {
      if (column == columnName) {
         return column_id;
      }
      column_id++;
   }
   throw std::runtime_error("column not found: " + columnName);
}
EXPORT void* rt_scan_source_init(runtime::ExecutionContext* executionContext, runtime::VarLen32 description) {

   nlohmann::json descr = nlohmann::json::parse(description.str());
   std::string tableName = descr["table"];
   auto table = (executionContext)->db->getTable(tableName);
   if (!table) {
      throw std::runtime_error("could not find table");
   }
   size_t numColumns = descr["columns"].size();
   auto tablePtr = new std::shared_ptr<arrow::Table>(table);
   size_t* res = new size_t[numColumns + 1];
   size_t i = 0;
   res[i++] = (size_t) tablePtr;
   for (std::string c : descr["columns"]) {
      res[i++] = get_column_id(tablePtr, c);
   }
   return res;
}

struct tableChunkIteratorStruct {
   arrow::TableBatchReader reader;
   std::shared_ptr<arrow::RecordBatch> curr_chunk;
   tableChunkIteratorStruct(arrow::Table& table) : reader(table), curr_chunk() {}
};

EXPORT tableChunkIteratorStruct* rt_table_chunk_iterator_init(std::shared_ptr<arrow::Table>* table) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto* tableChunkIterator = new tableChunkIteratorStruct(*(*table));
   if (tableChunkIterator->reader.ReadNext(&tableChunkIterator->curr_chunk) != arrow::Status::OK()) {
      tableChunkIterator->curr_chunk.reset();
   }
   return tableChunkIterator;
}
EXPORT tableChunkIteratorStruct* rt_table_chunk_iterator_next(tableChunkIteratorStruct* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if ((iterator)->reader.ReadNext(&(iterator)->curr_chunk) != arrow::Status::OK()) {
      (iterator)->curr_chunk.reset();
   }
   return iterator;
}
EXPORT arrow::RecordBatch* rt_table_chunk_iterator_curr(tableChunkIteratorStruct* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (iterator)->curr_chunk.get();
}

EXPORT bool rt_table_chunk_iterator_valid(tableChunkIteratorStruct* iterator) {
   auto valid = (iterator)->curr_chunk.operator bool();
   return valid;
}
EXPORT void rt_table_chunk_iterator_free(tableChunkIteratorStruct* iterator) {
   return delete iterator;
}
EXPORT uint64_t rt_table_chunk_num_rows(arrow::RecordBatch* tableChunk) {
   return (tableChunk)->num_rows();
}

EXPORT uint8_t* rt_table_chunk_get_column_buffer(arrow::RecordBatch* tableChunk, uint64_t columnId, uint64_t bufferId) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   static uint8_t alternative = 0b11111111;
   auto* buffer = (tableChunk)->column_data(columnId)->buffers[bufferId].get();
   if (buffer) {
      size_t offset = (tableChunk)->column_data(columnId)->offset;
      uint8_t* data = (uint8_t*) buffer->address();
      data += offset;
      return data;
   } else {
      return &alternative; //always return valid pointer to at least one byte filled with ones
   }
}
struct TuplePtrSize{
   uint8_t* data;
   size_t len;
};
EXPORT TuplePtrSize rt_table_chunk_get_raw_column_buffer(arrow::RecordBatch* tableChunk, uint64_t columnId, uint64_t bufferId) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   static uint8_t alternative = 0b11111111;
   auto* buffer = (tableChunk)->column_data(columnId)->buffers[bufferId].get();
   if (buffer) {
      uint8_t* data = (uint8_t*) buffer->address();
      size_t len = (tableChunk)->column_data(columnId)->buffers[bufferId].get()->size();

      return {data, len};
   } else {
      return {&alternative, 0}; //always return valid pointer to at least one byte filled with ones
   }
}
EXPORT uint64_t rt_table_chunk_get_column_offset(arrow::RecordBatch* tableChunk, uint64_t columnId) {
   return (tableChunk)->column_data(columnId)->offset;
}