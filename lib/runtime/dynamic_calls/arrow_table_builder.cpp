#include "runtime/TableBuilder.h"
#include "runtime/helpers.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/util/decimal.h>
#define EXPORT extern "C" __attribute__((visibility("default")))

runtime::TableBuilder* runtime::TableBuilder::create(VarLen32 schemaDescription) {
   return new TableBuilder(parseSchema(schemaDescription.str()));
}
void runtime::TableBuilder::destroy(TableBuilder* tb) {
   delete tb;
}
std::shared_ptr<arrow::Table>* runtime::TableBuilder::build() {
   flushBatch();
   std::shared_ptr<arrow::Table> table;
   auto st = arrow::Table::FromRecordBatches(schema, batches).Value(&table);
   if (st != arrow::Status::OK()) {
      throw std::runtime_error("could not create table:" + st.ToString());
   }
   return new std::shared_ptr<arrow::Table>(table);
}
void runtime::TableBuilder::addBool(bool isValid, bool value) {
   auto* typed_builder = GetBuilder<arrow::BooleanBuilder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      typed_builder->Append(value); //NOLINT (clang-diagnostic-unused-result)
   }
}

#define TABLE_BUILDER_ADD_PRIMITIVE(name, type)                                    \
   void runtime::TableBuilder::add##name(bool isValid, arrow::type ::c_type val) { \
      auto* typed_builder = GetBuilder<arrow::NumericBuilder<arrow::type>>();      \
      if (!isValid) {                                                              \
         typed_builder->AppendNull(); /*NOLINT (clang-diagnostic-unused-result)*/  \
      } else {                                                                     \
         typed_builder->Append(val); /*NOLINT (clang-diagnostic-unused-result)*/   \
      }                                                                            \
   }

TABLE_BUILDER_ADD_PRIMITIVE(Int8, Int8Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int16, Int16Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int32, Int32Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int64, Int64Type)
TABLE_BUILDER_ADD_PRIMITIVE(Float32, FloatType)
TABLE_BUILDER_ADD_PRIMITIVE(Float64, DoubleType)
void runtime::TableBuilder::addDecimal(bool isValid, __int128 value) {
   auto* typed_builder = GetBuilder<arrow::Decimal128Builder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(value >> 64, value));
      typed_builder->Append(decimalrep); //NOLINT (clang-diagnostic-unused-result)
   }
}
void runtime::TableBuilder::addBinary(bool isValid, runtime::VarLen32 string) {
   auto* typed_builder = GetBuilder<arrow::BinaryBuilder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      std::string str = (string).str();
      typed_builder->Append(string.getPtr(), string.getLen()); //NOLINT (clang-diagnostic-unused-result)
   }
}
void runtime::TableBuilder::nextRow() {
   assert(currColumn == schema->num_fields());
   currColumn = 0;
   if (current_batch_size >= max_batch_size) {
      flushBatch();
   }
   current_batch_size++;
}
