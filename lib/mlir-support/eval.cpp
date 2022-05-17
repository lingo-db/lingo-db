#include "mlir-support/eval.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_scalar.h>


namespace support::eval {
struct ArrowExpr : public Expr {
   arrow::compute::Expression expr;
   ArrowExpr(arrow::compute::Expression expr) : expr(expr) {}
};
arrow::compute::Expression unpack(const std::unique_ptr<expr>& expr) {
   return dynamic_cast<ArrowExpr&>(*expr).expr;
}
void init(){
   arrow::compute::GetFunctionRegistry();
}
std::unique_ptr<expr> pack(arrow::compute::Expression expr) {
   return std::make_unique<ArrowExpr>(expr);
}
std::optional<size_t> countResults(std::shared_ptr<arrow::RecordBatch> batch, std::unique_ptr<expr> filter) {
   if (!filter) return {};
   auto filterExpression = unpack(filter);
   auto boundCond = filterExpression.Bind(*batch->schema()).ValueOrDie();
   auto execBatch = arrow::compute::MakeExecBatch(*batch->schema(), batch);
   if (!execBatch.ok()) {
      throw std::runtime_error("Apache Arrow:" + execBatch->ToString());
   }
   arrow::Result<arrow::Datum> mask = arrow::compute::ExecuteScalarExpression(boundCond, execBatch.ValueUnsafe());
   if (!mask.ok()) {
      throw std::runtime_error("Apache Arrow:" + execBatch->ToString());
   }
   if (mask->is_scalar()) {
      return mask->scalar_as<arrow::BooleanScalar>().is_valid ? batch->num_rows() : 0;
   }
   size_t res = arrow::compute::internal::GetFilterOutputSize(*mask.ValueUnsafe().array(), arrow::compute::FilterOptions::NullSelectionBehavior::DROP);
   return res;
}
std::unique_ptr<expr> createAnd(const std::vector<std::unique_ptr<expr>>& expressions) {
   std::vector<arrow::compute::Expression> pureExpressions;
   for (const auto& e : expressions) {
      if (!e) return {};
      pureExpressions.push_back(unpack(e));
   }
   if (pureExpressions.size() == 1) return pack(pureExpressions[0]);
   assert(pureExpressions.size() > 1);
   auto res = arrow::compute::and_(pureExpressions);
   return pack(res);
}
std::unique_ptr<expr> createOr(const std::vector<std::unique_ptr<expr>>& expressions) {
   std::vector<arrow::compute::Expression> pureExpressions;
   for (const auto& e : expressions) {
      if (!e) return {};
      pureExpressions.push_back(unpack(e));
   }
   if (pureExpressions.size() == 1) return pack(pureExpressions[0]);
   assert(pureExpressions.size() > 1);
   auto res = arrow::compute::or_(pureExpressions);
   return pack(res);
}
std::unique_ptr<expr> createAttrRef(const std::string& str) {
   auto res= arrow::compute::field_ref(str);
   return pack(res);
}
std::unique_ptr<expr> createEq(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   if (!a || !b) return {};
   auto res = arrow::compute::equal(unpack(a), unpack(b));
   return pack(res);
}
std::unique_ptr<expr> createLt(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   if (!a || !b) return {};
   auto res = arrow::compute::less(unpack(a), unpack(b));
   return pack(res);
}
std::unique_ptr<expr> createGt(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   if (!a || !b) return {};
   auto res = arrow::compute::greater(unpack(a), unpack(b));
   return pack(res);
}
std::unique_ptr<expr> createLte(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   if (!a || !b) return {};
   auto res = arrow::compute::less_equal(unpack(a), unpack(b));
   return pack(res);
}
std::unique_ptr<expr> createGte(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   if (!a || !b) return {};
   auto res = arrow::compute::greater_equal(unpack(a), unpack(b));
   return pack(res);
}
std::unique_ptr<expr> createNot(std::unique_ptr<expr> a) {
   if (!a) return {};
   auto res = arrow::compute::not_(unpack(a));
   return pack(res);
}
std::unique_ptr<expr> createLiteral(std::variant<int64_t, double, std::string> parsed, std::tuple<arrow::Type::type, uint32_t, uint32_t> t) {
   auto [type, tp1, tp2] = t;
   switch (type) {
      case arrow::Type::type::INT8: return pack(arrow::compute::literal(std::make_shared<arrow::Int8Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::INT16: return pack(arrow::compute::literal(std::make_shared<arrow::Int16Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::INT32: return pack(arrow::compute::literal(std::make_shared<arrow::Int32Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::INT64: return pack(arrow::compute::literal(std::make_shared<arrow::Int64Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::UINT8: return pack(arrow::compute::literal(std::make_shared<arrow::UInt8Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::UINT16: return pack(arrow::compute::literal(std::make_shared<arrow::UInt16Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::UINT32: return pack(arrow::compute::literal(std::make_shared<arrow::UInt32Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::UINT64: return pack(arrow::compute::literal(std::make_shared<arrow::UInt64Scalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::BOOL: return pack(arrow::compute::literal(std::make_shared<arrow::BooleanScalar>(std::get<int64_t>(parsed))));
      case arrow::Type::type::INTERVAL_DAY_TIME: return {};
      case arrow::Type::type::INTERVAL_MONTHS: return {};
      case arrow::Type::type::INTERVAL_MONTH_DAY_NANO: return {};
      case arrow::Type::type::DATE32: return pack(arrow::compute::literal(std::make_shared<arrow::Date32Scalar>(std::get<int64_t>(parsed)/86400000000000ll)));
      case arrow::Type::type::DATE64: return pack(arrow::compute::literal(std::make_shared<arrow::Date64Scalar>(std::get<int64_t>(parsed)/1000000ll)));
      case arrow::Type::type::TIMESTAMP: return pack(arrow::compute::literal(std::make_shared<arrow::TimestampScalar>(std::get<int64_t>(parsed), static_cast<arrow::TimeUnit::type>(tp1))));
      case arrow::Type::type::HALF_FLOAT: return pack(arrow::compute::literal(std::make_shared<arrow::HalfFloatScalar>(std::get<double>(parsed))));
      case arrow::Type::type::FLOAT: return pack(arrow::compute::literal(std::make_shared<arrow::FloatScalar>(std::get<double>(parsed))));
      case arrow::Type::type::DOUBLE: return pack(arrow::compute::literal(std::make_shared<arrow::DoubleScalar>(std::get<double>(parsed))));
      case arrow::Type::type::DECIMAL128: {
         arrow::Decimal128 dec128;
         int32_t precision, scale;
         if (!arrow::Decimal128::FromString(std::get<std::string>(parsed), &dec128, &precision, &scale).ok()) {
            return {};
         }
         dec128 = dec128.Rescale(scale, tp2).ValueOrDie();
         auto decimalDataType = arrow::DecimalType::Make(arrow::Type::type::DECIMAL128, tp1, tp2).ValueOrDie();
         return pack(arrow::compute::literal(std::make_shared<arrow::Decimal128Scalar>(dec128, decimalDataType)));
      }
      case arrow::Type::type::STRING: return pack(arrow::compute::literal(std::make_shared<arrow::StringScalar>(std::get<std::string>(parsed))));
      case arrow::Type::type::FIXED_SIZE_BINARY: {
         std::shared_ptr<arrow::Buffer> bytes = arrow::AllocateResizableBuffer(tp1).ValueOrDie();
         int64_t val = std::get<int64_t>(parsed);
         memcpy(bytes->mutable_data(), &val, std::min(sizeof(val), static_cast<size_t>(tp1)));
         return pack(arrow::compute::literal(std::make_shared<arrow::FixedSizeBinaryScalar>(bytes, arrow::fixed_size_binary(tp1))));
      }
      default: return {};
   }
}

std::unique_ptr<expr> createLike(std::unique_ptr<expr> a, std::string like){
   if (!a) return {};
   auto options= std::make_shared<arrow::compute::MatchSubstringOptions>(like);
   std::vector<arrow::compute::Expression> args({unpack(a)});
   auto res = arrow::compute::call("match_like",args,options);
   return pack(res);
}
std::unique_ptr<expr> createInvalid() {
   return {};
}

} // end namespace support::eval
