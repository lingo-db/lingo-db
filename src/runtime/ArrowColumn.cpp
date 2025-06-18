#include "lingodb/runtime/ArrowColumn.h"

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/builder.h>
#include <lingodb/runtime/ExecutionContext.h>

using namespace lingodb::runtime;
namespace {
std::shared_ptr<arrow::Array> cast(std::shared_ptr<arrow::Array> array, std::shared_ptr<arrow::DataType> type) {
   auto arrayData = array->data();
   arrayData = arrow::ArrayData::Make(type, arrayData->length, arrayData->buffers, arrayData->child_data, arrayData->null_count, arrayData->offset);
   return arrow::MakeArray(arrayData);
}
//todo: avoid this
std::shared_ptr<arrow::DataType> physicalType(std::shared_ptr<arrow::DataType> t) {
   auto byteWidth = t->byte_width();
   //non-fixed-width, or fixed-width< 1 byte (e.g., boolean)
   if (byteWidth <= 0) {
      return t;
   } else {
      return arrow::fixed_size_binary(byteWidth);
   }
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
   auto* cb = new ArrowColumnBuilder(arrowType);
   getCurrentExecutionContext()->registerState({cb, [](void* ptr) { delete reinterpret_cast<ArrowColumnBuilder*>(ptr); }});
   return cb;
}

ArrowColumnBuilder::ArrowColumnBuilder(std::shared_ptr<arrow::DataType> type) : childBuilder(nullptr), type(type) {
   builderUnique = arrow::MakeBuilder(physicalType(type)).ValueOrDie();
   auto reserveOk = builderUnique->Reserve(20000).ok();
   assert(reserveOk);
   builder = builderUnique.get();
   if (type->id() == arrow::Type::LIST) {
      childBuilder = new ArrowColumnBuilder(reinterpret_cast<arrow::ListBuilder*>(builder)->value_builder());
   }
}
void ArrowColumnBuilder::next() {
   if (builderUnique) {
      //only do this for "root builder"
      numValues++;
      if (numValues > 20000) {
         auto array = builder->Finish().ValueOrDie();
         additionalArrays.push_back(cast(array, type));
         auto reserveOk = builderUnique->Reserve(20000).ok();
         assert(reserveOk);
         numValues = 0;
      }
   }
}
ArrowColumnBuilder::ArrowColumnBuilder(arrow::ArrayBuilder* valueBuilder) : type() {
   builder = valueBuilder;
   childBuilder = nullptr;
}
ArrowColumnBuilder* ArrowColumnBuilder::getChildBuilder() {
   if (!childBuilder) {
      throw std::runtime_error("child builder is null");
   }
   return childBuilder;
}
ArrowColumnBuilder::~ArrowColumnBuilder() {
   if (childBuilder) {
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

void ArrowColumnBuilder::addFixedSized(bool isValid, uint8_t* value) {
   next();
   auto* typedBuilder = reinterpret_cast<arrow::FixedSizeBinaryBuilder*>(builder);
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(value));
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
   auto* c = new ArrowColumn(column);
   getCurrentExecutionContext()->registerState({c, [](void* ptr) { delete reinterpret_cast<ArrowColumn*>(ptr); }});
   return c;
}
void ArrowColumnBuilder::merge(ArrowColumnBuilder* other) {
   auto array = other->builder->Finish().ValueOrDie();
   additionalArrays.push_back(cast(array, type));
   additionalArrays.insert(additionalArrays.end(), other->additionalArrays.begin(), other->additionalArrays.end());
}
