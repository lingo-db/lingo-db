#ifndef MLIR_SUPPORT_EVAL_H
#define MLIR_SUPPORT_EVAL_H

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <arrow/compute/type_fwd.h>
#include <arrow/type_fwd.h>
namespace support::eval {
using expr = arrow::compute::Expression;
expr createAttrRef(const std::string& str);
std::optional<expr> createLiteral(std::variant<int64_t, double, std::string> parsed, std::tuple<arrow::Type::type, uint32_t, uint32_t> type);
expr createAnd(const std::vector<expr>& expressions);
expr createOr(const std::vector<expr>& expressions);
expr createNot(expr a);
expr createEq(expr a, expr b);
expr createLt(expr a, expr b);
expr createGt(expr a, expr b);

std::optional<size_t> countResults(std::shared_ptr<arrow::RecordBatch> batch, expr& filter);
} // end namespace support::eval

#endif // MLIR_SUPPORT_EVAL_H