#include "lingodb/runtime/ArrowColumn.h"

#include "lingodb/runtime/ListRuntime.h"
#include "lingodb/runtime/StructRuntime.h"

#include <iomanip>
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
   if (typeDescr.starts_with("struct[") && typeDescr.ends_with(']')) {
      std::string inner = typeDescr.substr(7, typeDescr.size() - 8);
      std::vector<std::shared_ptr<arrow::Field>> fields;
      size_t start = 0;
      int nesting = 0;
      std::string currentField;
      for (size_t i = 0; i < inner.size(); ++i) {
         if (inner[i] == '[') nesting++;
         if (inner[i] == ']') nesting--;
         if (nesting == 0 && inner[i] == ',') {
            currentField = inner.substr(start, i - start);
            size_t colonPos = currentField.find(':');
            fields.push_back(arrow::field(currentField.substr(0, colonPos), parseType(currentField.substr(colonPos + 1))));
            start = i + 1;
         }
      }
      currentField = inner.substr(start);
      if (!currentField.empty()) {
         size_t colonPos = currentField.find(':');
         fields.push_back(arrow::field(currentField.substr(0, colonPos), parseType(currentField.substr(colonPos + 1))));
      }
      return arrow::struct_(fields);
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

void ArrowColumnBuilder::addList(bool isValid, List* list, VarLen32 type) {
   auto addListRecursive = [](auto&& self, arrow::ListBuilder* typedBuilder, class List* list, std::shared_ptr<arrow::DataType> arrowType) -> void {
      switch (arrowType->id()) {
         case arrow::Type::STRING: {
            auto* valueBuilder = reinterpret_cast<arrow::BinaryBuilder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               auto& varlen = *reinterpret_cast<VarLen32*>(ptr);
               handleStatus(valueBuilder->Append(varlen.getPtr(), varlen.getLen()));
            }
            break;
         }
         case arrow::Type::INT32: {
            auto* valueBuilder = reinterpret_cast<arrow::Int32Builder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               handleStatus(valueBuilder->Append(*reinterpret_cast<int32_t*>(ptr)));
            }
            break;
         }
         case arrow::Type::DECIMAL128: {
            auto* valueBuilder = reinterpret_cast<arrow::Decimal128Builder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               __int128 val = *reinterpret_cast<__int128*>(ptr);
               arrow::Decimal128 arrowVal(static_cast<int64_t>(val >> 64), static_cast<uint64_t>(val));
               handleStatus(valueBuilder->Append(arrowVal));
            }
            break;
         }
         case arrow::Type::DATE32: {
            auto* valueBuilder = reinterpret_cast<arrow::Date32Builder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               handleStatus(valueBuilder->Append(*reinterpret_cast<int32_t*>(ptr)));
            }
            break;
         }
         case arrow::Type::INT64: {
            auto* valueBuilder = reinterpret_cast<arrow::Int64Builder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               handleStatus(valueBuilder->Append(*reinterpret_cast<int64_t*>(ptr)));
            }
            break;
         }
         case arrow::Type::INT8: {
            auto* valueBuilder = reinterpret_cast<arrow::Int8Builder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               handleStatus(valueBuilder->Append(*reinterpret_cast<int8_t*>(ptr)));
            }
            break;
         }
         case arrow::Type::FIXED_SIZE_BINARY: {
            auto* valueBuilder = reinterpret_cast<arrow::FixedSizeBinaryBuilder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               handleStatus(valueBuilder->Append(ptr));
            }
            break;
         }
         case arrow::Type::LIST: {
            auto listType = std::static_pointer_cast<arrow::ListType>(arrowType);
            auto childElementType = listType->value_type();

            auto* innerListBuilder = reinterpret_cast<arrow::ListBuilder*>(typedBuilder->value_builder());
            for (size_t i = 0; i < list->size(); ++i) {
               uint8_t* ptr = list->at(i);
               auto* childList = *reinterpret_cast<List**>(ptr);
               handleStatus(innerListBuilder->Append());
               self(self, innerListBuilder, childList, childElementType);
            }
            break;
         }
         default: {
            throw std::runtime_error("unknown list element type");
         }
      }
   };
   next();
   auto* typedBuilder = reinterpret_cast<arrow::ListBuilder*>(builder);
   if (!isValid || !list) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      auto arrowType = parseType(type.str());
      handleStatus(typedBuilder->Append());
      addListRecursive(addListRecursive, typedBuilder, list, arrowType);
   }
}

namespace {

// Helper to determine the physical byte size
inline size_t getPhysicalByteSize(const std::shared_ptr<arrow::DataType>& type) {
   switch (type->id()) {
      case arrow::Type::INT8: return 1;
      case arrow::Type::INT32:
      case arrow::Type::DATE32: return 4;
      case arrow::Type::INT64: return 8;
      case arrow::Type::DECIMAL128: return 16;
      case arrow::Type::STRING: return sizeof(VarLen32);
      case arrow::Type::LIST:
      case arrow::Type::STRUCT: return sizeof(void*);
      case arrow::Type::FIXED_SIZE_BINARY: {
         auto fixedType = std::static_pointer_cast<arrow::FixedSizeBinaryType>(type);
         return fixedType->byte_width();
      }
      default: throw std::runtime_error("Unknown type size for struct offset calculation");
   }
}

// Helper to determine C++ memory alignment rules for the compiler
// Helper to determine C++ memory alignment rules for the compiler
// TODO Check
inline size_t getPhysicalAlignment(const std::shared_ptr<arrow::DataType>& type) {
   switch (type->id()) {
      case arrow::Type::INT8: return 1;
      case arrow::Type::INT32:
      case arrow::Type::DATE32: return 4;
      case arrow::Type::INT64:
      case arrow::Type::LIST:
      case arrow::Type::STRUCT: return 8;
      case arrow::Type::STRING: return 16;
      case arrow::Type::DECIMAL128: return 16;
      case arrow::Type::FIXED_SIZE_BINARY: return 1;
      default: return 8;
   }
}

} // namespace

void ArrowColumnBuilder::addStruct(bool isValid, Struct* structObj, VarLen32 type) {
   auto appendValue = [](auto&& self, arrow::ArrayBuilder* baseBuilder, std::shared_ptr<arrow::DataType> arrowType, uint8_t* ptr) -> void {
      if (!ptr) {
         handleStatus(baseBuilder->AppendNull());
         return;
      }

      switch (arrowType->id()) {
         case arrow::Type::STRING: {
            auto* typedBuilder = reinterpret_cast<arrow::BinaryBuilder*>(baseBuilder);
            auto& varlen = *reinterpret_cast<VarLen32*>(ptr);
            handleStatus(typedBuilder->Append(varlen.getPtr(), varlen.getLen()));
            break;
         }
         case arrow::Type::INT32: {
            auto* typedBuilder = reinterpret_cast<arrow::Int32Builder*>(baseBuilder);
            handleStatus(typedBuilder->Append(*reinterpret_cast<int32_t*>(ptr)));
            break;
         }
         case arrow::Type::DECIMAL128: {
            auto* typedBuilder = reinterpret_cast<arrow::Decimal128Builder*>(baseBuilder);
            __int128 val = *reinterpret_cast<__int128*>(ptr);
            arrow::Decimal128 arrowVal(static_cast<int64_t>(val >> 64), static_cast<uint64_t>(val));
            handleStatus(typedBuilder->Append(arrowVal));
            break;
         }
         case arrow::Type::DATE32: {
            auto* typedBuilder = reinterpret_cast<arrow::Date32Builder*>(baseBuilder);
            handleStatus(typedBuilder->Append(*reinterpret_cast<int32_t*>(ptr)));
            break;
         }
         case arrow::Type::INT64: {
            auto* typedBuilder = reinterpret_cast<arrow::Int64Builder*>(baseBuilder);
            handleStatus(typedBuilder->Append(*reinterpret_cast<int64_t*>(ptr)));
            break;
         }
         case arrow::Type::INT8: {
            auto* typedBuilder = reinterpret_cast<arrow::Int8Builder*>(baseBuilder);
            handleStatus(typedBuilder->Append(*reinterpret_cast<int8_t*>(ptr)));
            break;
         }
         case arrow::Type::FIXED_SIZE_BINARY: {
            auto* typedBuilder = reinterpret_cast<arrow::FixedSizeBinaryBuilder*>(baseBuilder);
            handleStatus(typedBuilder->Append(ptr));
            break;
         }
         case arrow::Type::LIST: {
            auto* typedBuilder = reinterpret_cast<arrow::ListBuilder*>(baseBuilder);
            auto listType = std::static_pointer_cast<arrow::ListType>(arrowType);
            auto childElementType = listType->value_type();
            auto* childList = *reinterpret_cast<List**>(ptr);

            handleStatus(typedBuilder->Append());
            auto* innerValueBuilder = typedBuilder->value_builder();
            for (size_t i = 0; i < childList->size(); ++i) {
               self(self, innerValueBuilder, childElementType, childList->at(i));
            }
            break;
         }
         case arrow::Type::STRUCT: {
            auto* typedBuilder = reinterpret_cast<arrow::StructBuilder*>(baseBuilder);
            auto nestedStructType = std::static_pointer_cast<arrow::StructType>(arrowType);
            auto* childStruct = *reinterpret_cast<Struct**>(ptr);

            handleStatus(typedBuilder->Append());

            size_t nestedOffset = 0;
            for (int i = 0; i < nestedStructType->num_fields(); ++i) {
               auto fieldType = nestedStructType->field(i)->type();
               auto* childBuilder = typedBuilder->child_builder(i).get();

               // Apply alignment padding for nested structs
               size_t alignment = getPhysicalAlignment(fieldType);
               size_t remainder = nestedOffset % alignment;
               if (remainder != 0) {
                  nestedOffset += (alignment - remainder);
               }

               uint8_t* nestedFieldPtr = childStruct ? childStruct->values.data() + nestedOffset : nullptr;
               self(self, childBuilder, fieldType, nestedFieldPtr);

               nestedOffset += getPhysicalByteSize(fieldType);
            }
            break;
         }
         default: {
            throw std::runtime_error("unknown struct field element type");
         }
      }
   };

   next();
   auto* typedBuilder = reinterpret_cast<arrow::StructBuilder*>(builder);

   if (!isValid || !structObj || structObj->values.empty()) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      auto arrowType = std::static_pointer_cast<arrow::StructType>(parseType(type.str()));
      handleStatus(typedBuilder->Append());

      size_t currentByteOffset = 0;
      uint8_t* basePointer = structObj->values.data();

      // Iterate through the Arrow schema and slice up the C++ vector sequentially
      for (int i = 0; i < arrowType->num_fields(); ++i) {
         auto fieldType = arrowType->field(i)->type();
         auto* childBuilder = typedBuilder->child_builder(i).get();

         // NEW: Apply alignment padding for parent struct
         size_t alignment = getPhysicalAlignment(fieldType);
         size_t remainder = currentByteOffset % alignment;
         if (remainder != 0) {
            currentByteOffset += (alignment - remainder);
         }

         // Calculate pointer for this specific field
         uint8_t* fieldPtr = basePointer + currentByteOffset;

         appendValue(appendValue, childBuilder, fieldType, fieldPtr);

         // Advance the offset for the next field based on actual type size
         currentByteOffset += getPhysicalByteSize(fieldType);
      }
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
