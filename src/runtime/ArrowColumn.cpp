#include "lingodb/runtime/ArrowColumn.h"

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/builder.h>

using namespace lingodb::runtime;
namespace {
std::shared_ptr<arrow::Array> cast(std::shared_ptr<arrow::Array> array, std::shared_ptr<arrow::DataType> type) {
   auto arrayData = array->data();
   arrayData = arrow::ArrayData::Make(type, arrayData->length, arrayData->buffers, arrayData->child_data, arrayData->null_count, arrayData->offset);
   return arrow::MakeArray(arrayData);
}
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
      return arrow::decimal128(p1, p2);
   } else if (name == "bool") {
      return arrow::boolean();
   }
   throw std::runtime_error("unknown type");
}
std::shared_ptr<arrow::DataType> parseType(std::string typeDescr) {
   if (typeDescr.starts_with("list[") && typeDescr.ends_with(']')) {
      auto elementType = parseType(typeDescr.substr(5, typeDescr.size() - 6));
      return arrow::list(elementType);
   }
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
   return createType(typeName, std::stoi(p1), std::stoi(p2));
}
void handleStatus(arrow::Status status) {
   if (!status.ok()) {
      throw std::runtime_error(status.ToString());
   }
}
} // end namespace
ArrowColumnBuilder* ArrowColumnBuilder::create(VarLen32 type) {
   auto arrowType = parseType(type.str());
   return new ArrowColumnBuilder(arrowType);
}

ArrowColumnBuilder::ArrowColumnBuilder(std::shared_ptr<arrow::DataType> type) : type(type) {
   builderUnique = arrow::MakeBuilder(GetPhysicalType(type)).ValueOrDie();
   auto reserveOk=builderUnique->Reserve(20000).ok();
   assert(reserveOk);
   builder=builderUnique.get();
   if (type->id() == arrow::Type::LIST){
      childBuilder = new ArrowColumnBuilder(reinterpret_cast<arrow::ListBuilder*>(builder)->value_builder());
   }
}
void ArrowColumnBuilder::next() {
   if(builderUnique) {
      //only do this for "root builder"
      numValues++;
      if(numValues>20000) {
         auto array = builder->Finish().ValueOrDie();
         additionalArrays.push_back(cast(array, type));
         auto reserveOk=builderUnique->Reserve(20000).ok();
         assert(reserveOk);
         numValues=0;
      }
   }
}
 ArrowColumnBuilder::ArrowColumnBuilder(arrow::ArrayBuilder* valueBuilder) :type(){

   builder=valueBuilder;
   childBuilder=nullptr;
}
ArrowColumnBuilder* ArrowColumnBuilder::getChildBuilder() {
   if(!childBuilder) {
      throw std::runtime_error("child builder is null");
   }
   return childBuilder;
}
 ArrowColumnBuilder::~ArrowColumnBuilder() {
   if(childBuilder) {
      delete childBuilder;
   }
}

void ArrowColumnBuilder::addBool(bool isValid, bool value) {
   next();
   auto* typedBuilder = reinterpret_cast<arrow::BooleanBuilder*>(builder);
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(value));
   }
}
#define COLUMN_BUILDER_ADD_PRIMITIVE(name, blubb)                                                 \
   void ArrowColumnBuilder::add##name(bool isValid, arrow::blubb ::c_type val) {                  \
      next();                                                                                     \
      auto* typedBuilder = reinterpret_cast<arrow::NumericBuilder<arrow::blubb>*>(builder); \
      if (!isValid) {                                                                             \
         handleStatus(typedBuilder->AppendNull()); /*NOLINT (clang-diagnostic-unused-result)*/    \
      } else {                                                                                    \
         handleStatus(typedBuilder->Append(val)); /*NOLINT (clang-diagnostic-unused-result)*/     \
      }                                                                                           \
   }

COLUMN_BUILDER_ADD_PRIMITIVE(Int8, Int8Type)
COLUMN_BUILDER_ADD_PRIMITIVE(Int16, Int16Type)
COLUMN_BUILDER_ADD_PRIMITIVE(Int32, Int32Type)
COLUMN_BUILDER_ADD_PRIMITIVE(Int64, Int64Type)
COLUMN_BUILDER_ADD_PRIMITIVE(Float32, FloatType)
COLUMN_BUILDER_ADD_PRIMITIVE(Float64, DoubleType)

void ArrowColumnBuilder::addDecimal(bool isValid, __int128 value) {
   next();
   auto* typedBuilder = reinterpret_cast<arrow::Decimal128Builder*>(builder);
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(value >> 64, value));
      handleStatus(typedBuilder->Append(decimalrep));
   }
}
void ArrowColumnBuilder::addBinary(bool isValid, lingodb::runtime::VarLen32 string) {
   next();
   auto* typedBuilder = reinterpret_cast<arrow::BinaryBuilder*>(builder);
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      std::string str = (string).str();
      handleStatus(typedBuilder->Append(string.getPtr(), string.getLen()));
   }
}
void ArrowColumnBuilder::addFixedSized(bool isValid, int64_t val) {
   next();
   auto* typedBuilder = reinterpret_cast<arrow::FixedSizeBinaryBuilder*>(builder);
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(reinterpret_cast<char*>(&val)));
   }
}
void ArrowColumnBuilder::addList(bool isValid) {
   next();
   auto* typedBuilder = reinterpret_cast<arrow::ListBuilder*>(builder);
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append());
   }
}
ArrowColumn* ArrowColumnBuilder::finish() {
   auto array = builder->Finish().ValueOrDie();
   additionalArrays.push_back(cast(array, type));
   auto column = std::make_shared<arrow::ChunkedArray>(additionalArrays, type);
   return new ArrowColumn(column);
}
void ArrowColumnBuilder::merge(ArrowColumnBuilder* other) {
   auto array = other->builder->Finish().ValueOrDie();
   additionalArrays.push_back(cast(array, type));
   additionalArrays.insert(additionalArrays.end(),other->additionalArrays.begin(),other->additionalArrays.end());
}
