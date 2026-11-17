#include "lingodb/runtime/ArrowColumn.h"
#include "lingodb/runtime/ArrowTypeDescription.h"

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/builder.h>
#include <arrow/type.h>
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

void handleStatus(arrow::Status status) {
   if (!status.ok()) {
      throw std::runtime_error(status.ToString());
   }
}
} // end namespace
ArrowColumnBuilder* ArrowColumnBuilder::create(VarLen32 type) {
   auto arrowType = parseArrowType(type.str());
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
   if (other->numValues) {
      auto array = other->builder->Finish().ValueOrDie();
      additionalArrays.push_back(cast(array, type));
      additionalArrays.insert(additionalArrays.end(), other->additionalArrays.begin(), other->additionalArrays.end());
   }
}
