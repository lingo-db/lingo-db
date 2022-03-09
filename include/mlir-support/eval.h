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
struct Expr {
   virtual ~Expr() = default;
};
using expr = Expr;
void init();
std::unique_ptr<expr> createInvalid();
std::unique_ptr<expr> createAttrRef(const std::string& str);
std::unique_ptr<expr> createLiteral(std::variant<int64_t, double, std::string> parsed, std::tuple<arrow::Type::type, uint32_t, uint32_t> type);
std::unique_ptr<expr> createAnd(const std::vector<std::unique_ptr<expr>>& expressions);
std::unique_ptr<expr> createOr(const std::vector<std::unique_ptr<expr>>& expressions);
std::unique_ptr<expr> createNot(std::unique_ptr<expr> a);
std::unique_ptr<expr> createEq(std::unique_ptr<expr> a, std::unique_ptr<expr> b);
std::unique_ptr<expr> createLt(std::unique_ptr<expr> a, std::unique_ptr<expr> b);
std::unique_ptr<expr> createLte(std::unique_ptr<expr> a, std::unique_ptr<expr> b);
std::unique_ptr<expr> createGte(std::unique_ptr<expr> a, std::unique_ptr<expr> b);
std::unique_ptr<expr> createGt(std::unique_ptr<expr> a, std::unique_ptr<expr> b);
std::unique_ptr<expr> createLike(std::unique_ptr<expr> a, std::string pattern);

std::optional<size_t> countResults(std::shared_ptr<arrow::RecordBatch> batch, std::unique_ptr<expr> filter);
} // end namespace support::eval

#endif // MLIR_SUPPORT_EVAL_H