#include "runtime/helpers.h"
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/util/decimal.h>

#define EXPORT extern "C" __attribute__((visibility("default")))

struct TableBuilder {
   static constexpr size_t max_batch_size = 2;
   std::shared_ptr<arrow::Schema> schema;
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   std::unique_ptr<arrow::RecordBatchBuilder> batchBuilder;
   size_t current_batch_size;
   TableBuilder(std::shared_ptr<arrow::Schema> schema) : schema(schema) {
      arrow::RecordBatchBuilder::Make(schema, arrow::default_memory_pool(), &batchBuilder); //NOLINT (clang-diagnostic-unused-result)
   }
   void flushBatch() {
      if (current_batch_size > 0) {
         std::shared_ptr<arrow::RecordBatch> recordBatch;
         batchBuilder->Flush(&recordBatch); //NOLINT (clang-diagnostic-unused-result)
         current_batch_size = 0;
         batches.push_back(recordBatch);
      }
   }
   void nextRow() {
      if (current_batch_size >= max_batch_size) {
         flushBatch();
      }
      current_batch_size++;
   }
   template <typename T>
   T* GetBuilderForColumn(int column) {
      return batchBuilder->GetFieldAs<T>(column);
   }
   std::shared_ptr<arrow::Table> build() {
      flushBatch();
      return arrow::Table::FromRecordBatches(schema, batches).ValueOrDie();
   }
};
EXPORT runtime::Pointer<TableBuilder> _mlir_ciface_arrow_create_table_builder(runtime::Pointer<std::shared_ptr<arrow::Schema>>* schema) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new TableBuilder((*schema).ref());
}

EXPORT void _mlir_ciface_table_builder_add_bool(runtime::Pointer<TableBuilder>* builder, int column, bool isNull, bool val) {
   auto* typed_builder = (*builder)->GetBuilderForColumn<arrow::BooleanBuilder>(column);
   if (isNull) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      typed_builder->Append(val); //NOLINT (clang-diagnostic-unused-result)
   }
}

#define TABLE_BUILDER_ADD_PRIMITIVE(name, type)                                                                                                    \
   EXPORT void _mlir_ciface_table_builder_add_##name(runtime::Pointer<TableBuilder>* builder, int column, bool isNull, arrow::type ::c_type val) { \
      auto* typed_builder = (*builder)->GetBuilderForColumn<arrow::NumericBuilder<arrow::type>>(column);                                           \
      if (isNull) {                                                                                                                                \
         typed_builder->AppendNull(); /*NOLINT (clang-diagnostic-unused-result)*/                                                                  \
      } else {                                                                                                                                     \
         typed_builder->Append(val); /*NOLINT (clang-diagnostic-unused-result)*/                                                                   \
      }                                                                                                                                            \
   }
TABLE_BUILDER_ADD_PRIMITIVE(int_8, Int8Type)
TABLE_BUILDER_ADD_PRIMITIVE(int_16, Int16Type)
TABLE_BUILDER_ADD_PRIMITIVE(int_32, Int32Type)
TABLE_BUILDER_ADD_PRIMITIVE(int_64, Int64Type)
TABLE_BUILDER_ADD_PRIMITIVE(float_32, FloatType)
TABLE_BUILDER_ADD_PRIMITIVE(float_64, DoubleType)
TABLE_BUILDER_ADD_PRIMITIVE(date_32, Date32Type)
TABLE_BUILDER_ADD_PRIMITIVE(date_64, Date64Type)
EXPORT void _mlir_ciface_table_builder_add_decimal(runtime::Pointer<TableBuilder>* builder, int column, bool isNull, int64_t low, int64_t high) {
   auto* typed_builder = (*builder)->GetBuilderForColumn<arrow::Decimal128Builder>(column);
   if (isNull) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
      typed_builder->Append(decimalrep); //NOLINT (clang-diagnostic-unused-result)
   }
}
EXPORT void _mlir_ciface_table_builder_add_binary(runtime::Pointer<TableBuilder>* builder, int column, bool isNull, runtime::String* string) {
   auto* typed_builder = (*builder)->GetBuilderForColumn<arrow::BinaryBuilder>(column);
   if (isNull) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      std::string str = (*string).str();
      typed_builder->Append(str.data(), str.size()); //NOLINT (clang-diagnostic-unused-result)
   }
}
EXPORT void _mlir_ciface_table_builder_finish_row(runtime::Pointer<TableBuilder>* builder) {
   (*builder)->nextRow();
}
EXPORT runtime::Pointer<std::shared_ptr<arrow::Table>> _mlir_ciface_table_builder_build(runtime::Pointer<TableBuilder>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new std::shared_ptr<arrow::Table>((*builder)->build());
}