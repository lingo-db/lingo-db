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

struct ColumnInfo {
   size_t offset;
   size_t validMultiplier;
   uint8_t* validBuffer;
   uint8_t* dataBuffer;
   uint8_t* varLenBuffer;
};
struct RecordBatchInfo {
   size_t numRows;
   ColumnInfo columnInfo[];
};
struct IteratorInfo {
   void* ptr;
   size_t columnIds[];
};

uint8_t* get_buffer(arrow::RecordBatch* batch, size_t columnId, size_t bufferId) {
   static uint8_t alternative = 0b11111111;
   if (batch->column_data(columnId)->buffers.size() > bufferId && batch->column_data(columnId)->buffers[bufferId]) {
      auto* buffer = batch->column_data(columnId)->buffers[bufferId].get();
      size_t offset = batch->column_data(columnId)->offset;
      return (uint8_t*) buffer->address();
   } else {
      return &alternative; //always return valid pointer to at least one byte filled with ones
   }
}
EXPORT void rt_access_record_batch(size_t numColumns, IteratorInfo* iteratorInfo, RecordBatchInfo* res, arrow::RecordBatch* recordBatch) {
   for (size_t i = 0; i < numColumns; i++) {
      auto colId = iteratorInfo->columnIds[i];
      ColumnInfo& colInfo = res->columnInfo[i];
      size_t off = recordBatch->column_data(colId)->offset;
      colInfo.offset = off;
      colInfo.validMultiplier = recordBatch->column_data(colId)->buffers[0] ? 1 : 0;
      colInfo.validBuffer = get_buffer(recordBatch, colId, 0);
      colInfo.dataBuffer = get_buffer(recordBatch, colId, 1);
      colInfo.varLenBuffer = get_buffer(recordBatch, colId, 2);
   }
   res->numRows = recordBatch->num_rows();
}
