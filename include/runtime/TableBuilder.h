#ifndef RUNTIME_TABLEBUILDER_H
#define RUNTIME_TABLEBUILDER_H
#include "runtime/helpers.h"

#include <cassert>

#include <arrow/table.h>
#include <arrow/table_builder.h>
namespace runtime {
class TableBuilder {
   static constexpr size_t maxBatchSize = 100000;
   std::shared_ptr<arrow::Schema> schema;
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   std::unique_ptr<arrow::RecordBatchBuilder> batchBuilder;
   size_t currentBatchSize = 0;
   size_t currColumn = 0;

   static std::shared_ptr<arrow::DataType> createType(std::string name, uint32_t p1, uint32_t p2) {
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
   static std::shared_ptr<arrow::Schema> parseSchema(std::string str) {
      if (str.empty()) {
         return std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{});
      }
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
   std::shared_ptr<arrow::Schema> lowerSchema(std::shared_ptr<arrow::Schema> schema) {
      std::vector<std::shared_ptr<arrow::Field>> fields;
      for (auto f : schema->fields()) {
         auto t = GetPhysicalType(f->type());
         if (arrow::is_fixed_size_binary(t->id())) {
            auto *fbt = dynamic_cast<arrow::FixedSizeBinaryType*>(t.get());
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
      for (int i = 0; i < recordBatch->num_columns(); i++) {
         columnData.push_back(arrow::ArrayData::Make(schema->field(i)->type(), recordBatch->column_data(i)->length, recordBatch->column_data(i)->buffers, recordBatch->column_data(i)->null_count, recordBatch->column_data(i)->offset));
      }
      return arrow::RecordBatch::Make(schema, recordBatch->num_rows(), columnData);
   }
   void flushBatch() {
      if (currentBatchSize > 0) {
         std::shared_ptr<arrow::RecordBatch> recordBatch;
         batchBuilder->Flush(&recordBatch); //NOLINT (clang-diagnostic-unused-result)
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
   void handleStatus(arrow::Status status){
      if(!status.ok()){
         throw std::runtime_error(status.ToString());
      }
   }

   public:
   static TableBuilder* create(VarLen32 schemaDescription);
   static void destroy(TableBuilder* tb);
   std::shared_ptr<arrow::Table>* build();

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
};
} // end namespace runtime
#endif //RUNTIME_TABLEBUILDER_H
