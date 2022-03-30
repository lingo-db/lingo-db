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

struct TableBuilder {
   static constexpr size_t max_batch_size = 100000;
   std::shared_ptr<arrow::Schema> schema;
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   std::unique_ptr<arrow::RecordBatchBuilder> batchBuilder;
   size_t current_batch_size = 0;
   size_t currColumn = 0;
   std::shared_ptr<arrow::Schema> lowerSchema(std::shared_ptr<arrow::Schema> schema) {
      std::vector<std::shared_ptr<arrow::Field>> fields;
      for (auto f : schema->fields()) {
         auto t = GetPhysicalType(f->type());
         if (arrow::is_fixed_size_binary(t->id())) {
            auto fbt = dynamic_cast<arrow::FixedSizeBinaryType*>(t.get());
            switch (fbt->byte_width()) {
               case 1: t = arrow::int8(); break;
               case 2: t = arrow::int16(); break;
               case 4: t = arrow::int32(); break;
               case 8: t = arrow::int64(); break;
            }
         }
         fields.push_back(std::make_shared<arrow::Field>(f->name(), t));
      }
      auto lowered = std::make_shared<arrow::Schema>(fields);
      //std::cout<<"lowered:"<<lowered->ToString()<<std::endl;
      return lowered;
   }
   TableBuilder(std::shared_ptr<arrow::Schema> schema) : schema(schema) {
      arrow::RecordBatchBuilder::Make(lowerSchema(schema), arrow::default_memory_pool(), &batchBuilder); //NOLINT (clang-diagnostic-unused-result)
   }
   std::shared_ptr<arrow::RecordBatch> convertBatch(std::shared_ptr<arrow::RecordBatch> recordBatch) {
      std::vector<std::shared_ptr<arrow::ArrayData>> columnData;
      for (size_t i = 0; i < recordBatch->num_columns(); i++) {
         columnData.push_back(arrow::ArrayData::Make(schema->field(i)->type(), recordBatch->column_data(i)->length, recordBatch->column_data(i)->buffers, recordBatch->column_data(i)->null_count, recordBatch->column_data(i)->offset));
      }
      return arrow::RecordBatch::Make(schema, recordBatch->num_rows(), columnData);
   }
   void flushBatch() {
      if (current_batch_size > 0) {
         std::shared_ptr<arrow::RecordBatch> recordBatch;
         batchBuilder->Flush(&recordBatch); //NOLINT (clang-diagnostic-unused-result)
         current_batch_size = 0;
         batches.push_back(convertBatch(recordBatch));
      }
   }
   void nextRow() {
      assert(currColumn == schema->num_fields());
      currColumn = 0;
      if (current_batch_size >= max_batch_size) {
         flushBatch();
      }
      current_batch_size++;
   }
   template <typename T>
   T* GetBuilder() {
      return batchBuilder->GetFieldAs<T>(currColumn++);
   }
   std::shared_ptr<arrow::Table> build() {
      flushBatch();
      std::shared_ptr<arrow::Table> table;
      auto st = arrow::Table::FromRecordBatches(schema, batches).Value(&table);
      if (st != arrow::Status::OK()) {
         throw std::runtime_error("could not create table:" + st.ToString());
      }
      return table;
   }
};

std::shared_ptr<arrow::DataType> createType(std::string name, uint32_t p1, uint32_t p2) {
   if (name == "int") {
      switch (p1) {
         case 8: return arrow::int8();
         case 16: return arrow::int16();
         case 32: return arrow::int32();
         case 64: return arrow::int64();
      }
   } else if (name == "uint") {
      switch (p1) {
         case 8: return arrow::uint8();
         case 16: return arrow::uint16();
         case 32: return arrow::uint32();
         case 64: return arrow::uint64();
      }
   } else if (name == "float") {
      switch (p1) {
         case 16: return arrow::float16();
         case 32: return arrow::float32();
         case 64: return arrow::float64();
      }
   } else if (name == "string") {
      return arrow::utf8();
   } else if (name == "fixed_sized") {
      return arrow::fixed_size_binary(p1);
   } else if (name == "date") {
      return p1 == 32 ? arrow::date32() : arrow::date64();
   } else if (name == "interval_months") {
      return arrow::month_interval();
   } else if (name == "interval_daytime") {
      return arrow::day_time_interval();
   } else if (name == "timestamp") {
      return arrow::timestamp(static_cast<arrow::TimeUnit::type>(p1));
   } else if (name == "decimal") {
      return arrow::decimal(p1, p2);
   } else if (name == "bool") {
      return arrow::boolean();
   }
   throw std::runtime_error("unknown type");
}
std::shared_ptr<arrow::Schema> parseSchema(std::string str) {
   std::vector<std::shared_ptr<arrow::Field>> fields;

   str.erase(std::remove_if(str.begin(), str.end(), [](char c) { return c == ' '; }), str.end());
   auto parseEntry = [&fields](std::string token) {
      size_t colonPos = token.find(":");
      if (colonPos == std::string::npos) throw std::runtime_error("expected ':'");
      std::string colName = token.substr(0, colonPos);
      std::string typeDescr = token.substr(colonPos + 1);
      size_t lParamPos = typeDescr.find("[");
      std::string p1 = "0";
      std::string p2 = "0";
      std::string typeName = typeDescr;
      if (lParamPos != std::string ::npos) {
         typeName = typeDescr.substr(0, lParamPos);
         assert(typeDescr.ends_with(']'));
         std::string paramString = typeDescr.substr(lParamPos + 1, typeDescr.size() - lParamPos - 2);
         size_t commaPos = paramString.find(",");
         if (commaPos == std::string::npos) {
            p1 = paramString;
         } else {
            p1 = paramString.substr(0, commaPos);
            p2 = paramString.substr(commaPos + 1);
         }
      }
      fields.push_back(std::make_shared<arrow::Field>(colName, createType(typeName, std::stoi(p1), std::stoi(p2))));
   };
   size_t pos = 0;
   std::string token;
   while ((pos = str.find(";")) != std::string::npos) {
      token = str.substr(0, pos);
      str.erase(0, pos + 1);
      parseEntry(token);
   }
   parseEntry(str);
   return std::make_shared<arrow::Schema>(fields);
}
EXPORT TableBuilder* rt_arrow_create_table_builder(runtime::VarLen32 schema) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   std::string str=schema.str();
   return new TableBuilder(parseSchema(str));
}

EXPORT void rt_table_builder_add_bool(TableBuilder* builder, bool isValid, bool val) {
   auto* typed_builder = (builder)->GetBuilder<arrow::BooleanBuilder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      typed_builder->Append(val); //NOLINT (clang-diagnostic-unused-result)
   }
}

#define TABLE_BUILDER_ADD_PRIMITIVE(name, type)                                                             \
   EXPORT void rt_table_builder_add_##name(TableBuilder* builder, bool isValid, arrow::type ::c_type val) { \
      auto* typed_builder = (builder)->GetBuilder<arrow::NumericBuilder<arrow::type>>();                    \
      if (!isValid) {                                                                                       \
         typed_builder->AppendNull(); /*NOLINT (clang-diagnostic-unused-result)*/                           \
      } else {                                                                                              \
         typed_builder->Append(val); /*NOLINT (clang-diagnostic-unused-result)*/                            \
      }                                                                                                     \
   }
EXPORT void rt_table_builder_add_date_32(TableBuilder* builder, bool isValid, int64_t val) {
   val = val / 86400000000000;
   auto* typed_builder = (builder)->GetBuilder<arrow::NumericBuilder<arrow::Int32Type>>();
   if (!isValid) {
      typed_builder->AppendNull(); /*NOLINT (clang-diagnostic-unused-result)*/
   } else {
      typed_builder->Append(val); /*NOLINT (clang-diagnostic-unused-result)*/
   }
}
EXPORT void rt_table_builder_add_date_64(TableBuilder* builder, bool isValid, int64_t val) {
   val = val / 1000000;
   auto* typed_builder = (builder)->GetBuilder<arrow::NumericBuilder<arrow::Int64Type>>();
   if (!isValid) {
      typed_builder->AppendNull(); /*NOLINT (clang-diagnostic-unused-result)*/
   } else {
      typed_builder->Append(val); /*NOLINT (clang-diagnostic-unused-result)*/
   }
}
TABLE_BUILDER_ADD_PRIMITIVE(int_8, Int8Type)
TABLE_BUILDER_ADD_PRIMITIVE(int_16, Int16Type)
TABLE_BUILDER_ADD_PRIMITIVE(int_32, Int32Type)
TABLE_BUILDER_ADD_PRIMITIVE(int_64, Int64Type)
TABLE_BUILDER_ADD_PRIMITIVE(float_32, FloatType)
TABLE_BUILDER_ADD_PRIMITIVE(float_64, DoubleType)
EXPORT void rt_table_builder_add_decimal(TableBuilder* builder, bool isValid, int64_t low, int64_t high) {
   auto* typed_builder = (builder)->GetBuilder<arrow::Decimal128Builder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
      typed_builder->Append(decimalrep); //NOLINT (clang-diagnostic-unused-result)
   }
}
EXPORT void rt_table_builder_add_small_decimal(TableBuilder* builder, bool isValid, int64_t low) {
   __int128 total = low;
   int64_t high = total >> 64;
   auto* typed_builder = (builder)->GetBuilder<arrow::Decimal128Builder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
      typed_builder->Append(decimalrep); //NOLINT (clang-diagnostic-unused-result)
   }
}
EXPORT void rt_table_builder_add_binary(TableBuilder* builder, bool isValid, runtime::VarLen32 string) {
   auto* typed_builder = (builder)->GetBuilder<arrow::BinaryBuilder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      std::string str = (string).str();
      typed_builder->Append(string.getPtr(), string.getLen()); //NOLINT (clang-diagnostic-unused-result)
   }
}
EXPORT void rt_table_builder_add_fixed_binary(TableBuilder* builder, bool isValid, uint64_t x) {
   auto* typed_builder = (builder)->GetBuilder<arrow::FixedSizeBinaryBuilder>();
   if (!isValid) {
      typed_builder->AppendNull(); //NOLINT (clang-diagnostic-unused-result)
   } else {
      typed_builder->Append((char*) &x);
   }
}
EXPORT void rt_table_builder_finish_row(TableBuilder* builder) {
   (builder)->nextRow();
}
EXPORT std::shared_ptr<arrow::Table>* rt_table_builder_build(TableBuilder* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new std::shared_ptr<arrow::Table>((builder)->build());
}