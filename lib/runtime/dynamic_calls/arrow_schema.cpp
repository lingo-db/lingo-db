#include "runtime/helpers.h"
#include <arrow/array.h>

EXPORT runtime::Pointer<std::shared_ptr<arrow::DataType>> _mlir_ciface_arrow_type2(uint32_t typeVal, uint32_t p1, uint32_t p2) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto* ptr = new std::shared_ptr<arrow::DataType>;
   arrow::Type::type type = static_cast<arrow::Type::type>(typeVal);
   switch (type) {
      case arrow::Type::BOOL: *ptr = arrow::boolean(); break;
      case arrow::Type::INT8: *ptr = arrow::int8(); break;
      case arrow::Type::INT16: *ptr = arrow::int16(); break;
      case arrow::Type::INT32: *ptr = arrow::int32(); break;
      case arrow::Type::INT64: *ptr = arrow::int64(); break;
      case arrow::Type::DECIMAL128: *ptr = arrow::decimal128(p1, p2); break;
      case arrow::Type::HALF_FLOAT: *ptr = arrow::float16(); break;
      case arrow::Type::FLOAT: *ptr = arrow::float32(); break;
      case arrow::Type::DOUBLE: *ptr = arrow::float64(); break;
      case arrow::Type::STRING: *ptr = arrow::utf8(); break;
      case arrow::Type::DATE32: *ptr = arrow::date32(); break;
      case arrow::Type::DATE64: *ptr = arrow::date64(); break;

      default: throw std::runtime_error("unknown arrow type!!");
   }
   return ptr;
}

EXPORT runtime::Pointer<std::shared_ptr<arrow::DataType>> _mlir_ciface_arrow_type1(uint32_t typeVal, uint32_t p1) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return _mlir_ciface_arrow_type2(typeVal, 0, 0);
}
EXPORT runtime::Pointer<std::shared_ptr<arrow::DataType>> _mlir_ciface_arrow_type(uint32_t typeVal) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return _mlir_ciface_arrow_type1(typeVal, 0);
}
struct SchemaBuilder {
   std::vector<std::shared_ptr<arrow::Field>> fields;
   std::shared_ptr<arrow::Schema>* build() {
      return new std::shared_ptr<arrow::Schema>(std::make_shared<arrow::Schema>(fields));
   }
};
EXPORT runtime::Pointer<SchemaBuilder> _mlir_ciface_arrow_schema_create_builder() { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new SchemaBuilder;
}
EXPORT void _mlir_ciface_arrow_schema_add_field(runtime::Pointer<SchemaBuilder>* builder, runtime::Pointer<std::shared_ptr<arrow::DataType>>* datatype, bool nullable, runtime::String* columnName) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (*builder)->fields.push_back(std::make_shared<arrow::Field>(columnName->str(), (*datatype).ref(), nullable));
}
EXPORT runtime::Pointer<std::shared_ptr<arrow::Schema>> _mlir_ciface_arrow_schema_build(runtime::Pointer<SchemaBuilder>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->build();
}