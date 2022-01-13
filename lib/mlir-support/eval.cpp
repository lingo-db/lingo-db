#include "mlir-support/eval.h"
#include <arrow/api.h>
#include <arrow/compute/api.h>
namespace support::eval {
std::optional<size_t> countResults(std::shared_ptr<arrow::RecordBatch> batch, expr& filter) {
   auto boundCond = filter.Bind(*batch->schema()).ValueOrDie();
   auto res2 = arrow::compute::ExecuteScalarExpression(boundCond, arrow::compute::MakeExecBatch(*batch->schema(), batch).ValueOrDie()).ValueOrDie();

   return arrow::compute::Count(arrow::compute::Filter(res2, res2).ValueOrDie()).ValueOrDie().scalar_as<arrow::Int64Scalar>().value;
}
expr createAnd(const std::vector<expr>& expressions) {
   return arrow::compute::and_(expressions);
}
expr createOr(const std::vector<expr>& expressions) {
   return arrow::compute::or_(expressions);
}
expr createAttrRef(const std::string& str) {
   return arrow::compute::field_ref(str);
}
expr createEq(expr a, expr b) {
   return arrow::compute::equal(a, b);
}
expr createLt(expr a, expr b) {
   return arrow::compute::less(a, b);
}
expr createGt(expr a, expr b) {
   return arrow::compute::greater(a, b);
}
expr createNot(expr a) {
   return arrow::compute::not_(a);
}
std::optional<expr> createLiteral(std::variant<int64_t, double, std::string> parsed, std::tuple<arrow::Type::type, uint32_t, uint32_t> t) {
   auto [type, tp1, tp2] = t;
   switch (type) {
      case arrow::Type::type::INT8: return arrow::compute::literal(std::make_shared<arrow::Int8Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::INT16: return arrow::compute::literal(std::make_shared<arrow::Int16Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::INT32: return arrow::compute::literal(std::make_shared<arrow::Int32Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::INT64: return arrow::compute::literal(std::make_shared<arrow::Int64Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::UINT8: return arrow::compute::literal(std::make_shared<arrow::UInt8Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::UINT16: return arrow::compute::literal(std::make_shared<arrow::UInt16Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::UINT32: return arrow::compute::literal(std::make_shared<arrow::UInt32Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::UINT64: return arrow::compute::literal(std::make_shared<arrow::UInt64Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::BOOL: return arrow::compute::literal(std::make_shared<arrow::BooleanScalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::INTERVAL_DAY_TIME: return {};
      case arrow::Type::type::INTERVAL_MONTHS: return {};
      case arrow::Type::type::INTERVAL_MONTH_DAY_NANO: return {};
      case arrow::Type::type::DATE32: return arrow::compute::literal(std::make_shared<arrow::Date32Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::DATE64: return arrow::compute::literal(std::make_shared<arrow::Date64Scalar>( std::get<int64_t>(parsed)));
      case arrow::Type::type::TIMESTAMP: return arrow::compute::literal(std::make_shared<arrow::TimestampScalar>( std::get<int64_t>(parsed), static_cast<arrow::TimeUnit::type>(tp1)));
      case arrow::Type::type::HALF_FLOAT: return arrow::compute::literal(std::make_shared<arrow::HalfFloatScalar>(std::get<double>(parsed)));
      case arrow::Type::type::FLOAT: return arrow::compute::literal(std::make_shared<arrow::FloatScalar>(std::get<double>(parsed)));
      case arrow::Type::type::DOUBLE: return arrow::compute::literal(std::make_shared<arrow::DoubleScalar>(std::get<double>(parsed)));
      case arrow::Type::type::DECIMAL128: {
         arrow::Decimal128 dec128;
         int32_t precision, scale;
         if (!arrow::Decimal128::FromString(std::get<std::string>(parsed), &dec128, &precision, &scale).ok()) {
            return {};
         }
         dec128 = dec128.Rescale(scale, tp2).ValueOrDie();
         auto decimalDataType = arrow::DecimalType::Make(arrow::Type::type::DECIMAL128, tp1, tp2).ValueOrDie();
         return arrow::compute::literal(std::make_shared<arrow::Decimal128Scalar>(dec128, decimalDataType));
      }
      case arrow::Type::type::STRING: return arrow::compute::literal(std::make_shared<arrow::StringScalar>(std::get<std::string>(parsed)));
      case arrow::Type::type::FIXED_SIZE_BINARY: {
         std::shared_ptr<arrow::Buffer> bytes = arrow::AllocateResizableBuffer(tp1).ValueOrDie();
         int64_t val = std::get<int64_t>(parsed);
         memcpy(bytes->mutable_data(), &val, std::min(sizeof(val), static_cast<size_t>(tp1)));
         return arrow::compute::literal(std::make_shared<arrow::FixedSizeBinaryScalar>(bytes, arrow::fixed_size_binary(tp1)));
      }
      default: return {};
   }
}

} // end namespace support::eval