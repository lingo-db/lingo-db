#include "runtime/TableBuilder.h"
#include "runtime/helpers.h"
#include <iostream>
#include <string>

#include "utility/Tracer.h"
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/util/decimal.h>

namespace {
static utility::Tracer::Event tableBuilderMerge("TableBuilder", "merge");
} // end namespace
struct TableBuilder {
   static constexpr size_t maxBatchSize = 100000;
   std::shared_ptr<arrow::Schema> schema;
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   std::unique_ptr<arrow::RecordBatchBuilder> batchBuilder;
   size_t currentBatchSize = 0;
   size_t currColumn = 0;


   std::shared_ptr<arrow::Schema> lowerSchema(std::shared_ptr<arrow::Schema> schema) {
      std::vector<std::shared_ptr<arrow::Field>> fields;
      for (auto f : schema->fields()) {
         auto t = GetPhysicalType(f->type());
         fields.push_back(std::make_shared<arrow::Field>(f->name(), t));
      }
      auto lowered = std::make_shared<arrow::Schema>(fields);
      //std::cout<<"lowered:"<<lowered->ToString()<<std::endl;
      return lowered;
   }
   TableBuilder(std::shared_ptr<arrow::Schema> schema) : schema(schema) {
      batchBuilder=arrow::RecordBatchBuilder::Make(lowerSchema(schema), arrow::default_memory_pool()).ValueOrDie();
   }
   std::shared_ptr<arrow::RecordBatch> convertBatch(std::shared_ptr<arrow::RecordBatch> recordBatch) {
      std::vector<std::shared_ptr<arrow::ArrayData>> columnData;
      for (int i = 0; i < recordBatch->num_columns(); i++) {
         columnData.push_back(arrow::ArrayData::Make(schema->field(i)->type(), recordBatch->column_data(i)->length, recordBatch->column_data(i)->buffers, recordBatch->column_data(i)->null_count, recordBatch->column_data(i)->offset));
      }
      return arrow::RecordBatch::Make(schema, recordBatch->num_rows(), columnData);
   }
   void flushBatch() {
      if (currentBatchSize > 0) {
         std::shared_ptr<arrow::RecordBatch> recordBatch;
         recordBatch=batchBuilder->Flush(true).ValueOrDie(); //NOLINT (clang-diagnostic-unused-result)
         currentBatchSize = 0;
         batches.push_back(convertBatch(recordBatch));
      }
   }
   template <typename T>
   T* getBuilder() {
      auto ptr = batchBuilder->GetFieldAs<T>(currColumn++);
      assert(ptr != nullptr);
      return ptr;
   }
   void handleStatus(arrow::Status status) {
      if (!status.ok()) {
         throw std::runtime_error(status.ToString());
      }
   }

   public:
   std::shared_ptr<arrow::Table> build();

   void addBool(bool isValid, bool value);
   void addInt8(bool isValid, int8_t);
   void addInt16(bool isValid, int16_t);
   void addInt32(bool isValid, int32_t);
   void addInt64(bool isValid, int64_t);
   void addFloat32(bool isValid, float);
   void addFloat64(bool isValid, double);
   void addDecimal(bool isValid, __int128);
   void addFixedSized(bool isValid, int64_t);
   void addBinary(bool isValid, runtime::VarLen32);
   void nextRow();
   void merge(TableBuilder* other) {
      other->flushBatch();
      batches.insert(batches.end(), other->batches.begin(), other->batches.end());
      other->batches.clear();
   }
};

#define EXPORT extern "C" __attribute__((visibility("default")))


std::shared_ptr<arrow::Table> TableBuilder::build() {
   flushBatch();
   std::shared_ptr<arrow::Table> table;
   auto st = arrow::Table::FromRecordBatches(schema, batches).Value(&table);
   if (!st.ok()) {
      throw std::runtime_error("could not create table:" + st.ToString());
   }
   return table;
}
void TableBuilder::addBool(bool isValid, bool value) {
   auto* typedBuilder = getBuilder<arrow::BooleanBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(value));
   }
}

#define TABLE_BUILDER_ADD_PRIMITIVE(name, type)                                                \
   void TableBuilder::add##name(bool isValid, arrow::type ::c_type val) {                      \
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

void TableBuilder::addDecimal(bool isValid, __int128 value) {
   auto* typedBuilder = getBuilder<arrow::Decimal128Builder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(value >> 64, value));
      handleStatus(typedBuilder->Append(decimalrep));
   }
}
void TableBuilder::addBinary(bool isValid, runtime::VarLen32 string) {
   auto* typedBuilder = getBuilder<arrow::BinaryBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      std::string str = (string).str();
      handleStatus(typedBuilder->Append(string.getPtr(), string.getLen()));
   }
}
void TableBuilder::addFixedSized(bool isValid, int64_t val) {
   auto* typedBuilder = getBuilder<arrow::FixedSizeBinaryBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(reinterpret_cast<char*>(&val)));
   }
}
void TableBuilder::nextRow() {
   assert(currColumn == static_cast<size_t>(schema->num_fields()));
   currColumn = 0;
   if (currentBatchSize >= maxBatchSize) {
      flushBatch();
   }
   currentBatchSize++;
}

#define RESULT_TABLE_FORWARD(name, type)                     \
   void runtime::ResultTable::name(bool isValid, type val) { \
      builder->name(isValid, val);                           \
   }

RESULT_TABLE_FORWARD(addBool, bool);
RESULT_TABLE_FORWARD(addInt8, int8_t);
RESULT_TABLE_FORWARD(addInt16, int16_t);
RESULT_TABLE_FORWARD(addInt32, int32_t);
RESULT_TABLE_FORWARD(addInt64, int64_t);
RESULT_TABLE_FORWARD(addFloat32, float);
RESULT_TABLE_FORWARD(addFloat64, double);
RESULT_TABLE_FORWARD(addDecimal, __int128);
RESULT_TABLE_FORWARD(addBinary, runtime::VarLen32);
RESULT_TABLE_FORWARD(addFixedSized, int64_t);
void runtime::ResultTable::nextRow() {
   builder->nextRow();
}
runtime::ResultTable* runtime::ResultTable::create(runtime::ExecutionContext* executionContext, runtime::ArrowSchema* schema) {
   ResultTable* resultTable = new ResultTable;
   resultTable->builder = new TableBuilder(schema->getSchema());
   executionContext->registerState({resultTable, [](void* ptr) { delete reinterpret_cast<ResultTable*>(ptr); }});
   return resultTable;
}
std::shared_ptr<arrow::Table> runtime::ResultTable::get() {
   if (resultTable) {
      return resultTable;
   } else {
      resultTable = builder->build();
   }
   return resultTable;
}

runtime::ResultTable* runtime::ResultTable::merge(runtime::ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(tableBuilderMerge);
   ResultTable* first = nullptr;
   for (auto* ptr : threadLocal->getTls()) {
      auto* current = reinterpret_cast<ResultTable*>(ptr);
      if (!first) {
         first = current;
      } else {
         first->builder->merge(current->builder);
      }
   }
   trace.stop();
   return first;
}