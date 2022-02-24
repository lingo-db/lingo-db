#include "mlir/Conversion/DBToArrowStd/ArrowTypes.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
std::tuple<arrow::Type::type, uint32_t, uint32_t> mlir::db::codegen::convertTypeToArrow(mlir::Type type) {
   arrow::Type::type typeConstant=arrow::Type::type::NA;
   uint32_t param1 = 0, param2 = 0;
   if (auto intType = type.dyn_cast_or_null<mlir::db::IntType>()) {
      switch (intType.getWidth()) {
         case 8: typeConstant = arrow::Type::type::INT8; break;
         case 16: typeConstant = arrow::Type::type::INT16; break;
         case 32: typeConstant = arrow::Type::type::INT32; break;
         case 64: typeConstant = arrow::Type::type::INT64; break;
      }
   } else if (auto intType = type.dyn_cast_or_null<mlir::db::UIntType>()) {
      switch (intType.getWidth()) {
         case 8: typeConstant = arrow::Type::type::UINT8; break;
         case 16: typeConstant = arrow::Type::type::UINT16; break;
         case 32: typeConstant = arrow::Type::type::UINT32; break;
         case 64: typeConstant = arrow::Type::type::UINT64; break;
      }
   } else if (isIntegerType(type,1)) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      typeConstant = arrow::Type::type::DECIMAL128;
      param1 = decimalType.getP();
      param2 = decimalType.getS();
   } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
      switch (floatType.getWidth()) {
         case 16: typeConstant = arrow::Type::type::HALF_FLOAT; break;
         case 32: typeConstant = arrow::Type::type::FLOAT; break;
         case 64: typeConstant = arrow::Type::type::DOUBLE; break;
      }
   } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
      typeConstant = arrow::Type::type::STRING;
   } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
      if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
         typeConstant = arrow::Type::type::DATE32;
      } else {
         typeConstant = arrow::Type::type::DATE64;
      }
   } else if (auto charType = type.dyn_cast_or_null<mlir::db::CharType>()) {
      typeConstant = arrow::Type::type::FIXED_SIZE_BINARY;
      param1 = charType.getBytes();
   } else if (auto intervalType=type.dyn_cast_or_null<mlir::db::IntervalType>()){
      if(intervalType.getUnit()==mlir::db::IntervalUnitAttr::months){
         typeConstant=arrow::Type::type::INTERVAL_MONTHS;
      }else{
         typeConstant=arrow::Type::type::INTERVAL_DAY_TIME;
      }
   } else if (auto timestampType =type.dyn_cast_or_null<mlir::db::TimestampType>()){
      typeConstant=arrow::Type::type::TIMESTAMP;
      param1=static_cast<uint32_t>(timestampType.getUnit());
   }
   assert(typeConstant!=arrow::Type::type::NA);
   return {typeConstant, param1, param2};
}