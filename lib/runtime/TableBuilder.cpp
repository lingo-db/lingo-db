#include "runtime/TableBuilder.h"
#include "runtime/helpers.h"
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
   if (!st.ok()) {
      throw std::runtime_error("could not create table:" + st.ToString());
   }
   return new std::shared_ptr<arrow::Table>(table);
}
void runtime::TableBuilder::addBool(bool isValid, bool value) {
   auto* typedBuilder = getBuilder<arrow::BooleanBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(value));
   }
}

#define TABLE_BUILDER_ADD_PRIMITIVE(name, type)                                                \
   void runtime::TableBuilder::add##name(bool isValid, arrow::type ::c_type val) {             \
      auto* typedBuilder = getBuilder<arrow::NumericBuilder<arrow::type>>();                   \
      if (!isValid) {                                                                          \
         handleStatus(typedBuilder->AppendNull()); /*NOLINT (clang-diagnostic-unused-result)*/ \
      } else {                                                                                 \
         handleStatus(typedBuilder->Append(val)); /*NOLINT (clang-diagnostic-unused-result)*/  \
      }                                                                                        \
   }

TABLE_BUILDER_ADD_PRIMITIVE(Int8, Int8Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int16, Int16Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int32, Int32Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int64, Int64Type)
TABLE_BUILDER_ADD_PRIMITIVE(Float32, FloatType)
TABLE_BUILDER_ADD_PRIMITIVE(Float64, DoubleType)

void runtime::TableBuilder::addDecimal(bool isValid, __int128 value) {
   auto* typedBuilder = getBuilder<arrow::Decimal128Builder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(value >> 64, value));
      handleStatus(typedBuilder->Append(decimalrep));
   }
}
void runtime::TableBuilder::addBinary(bool isValid, runtime::VarLen32 string) {
   auto* typedBuilder = getBuilder<arrow::BinaryBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      std::string str = (string).str();
      handleStatus(typedBuilder->Append(string.getPtr(), string.getLen()));
   }
}
void runtime::TableBuilder::addFixedSized(bool isValid, int64_t val) {
   auto* typedBuilder = getBuilder<arrow::FixedSizeBinaryBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(reinterpret_cast<char*>(&val)));
   }
}
void runtime::TableBuilder::nextRow() {
   assert(currColumn == static_cast<size_t>(schema->num_fields()));
   currColumn = 0;
   if (currentBatchSize >= maxBatchSize) {
      flushBatch();
   }
   currentBatchSize++;
}
