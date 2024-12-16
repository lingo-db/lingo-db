#include "lingodb/compiler/mlir-support/tostring.h"
#include <arrow/util/decimal.h>
#include <arrow/vendored/datetime.h>
namespace {
arrow_vendored::date::sys_days epoch = arrow_vendored::date::sys_days{arrow_vendored::date::jan / 1 / 1970};
} // namespace

std::string lingodb::compiler::support::decimalToString(uint64_t low, uint64_t high, int32_t scale) {
   arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
   return decimalrep.ToString(scale);
}
std::string lingodb::compiler::support::dateToString(uint64_t date) {
   return arrow_vendored::date::format("%F", epoch + std::chrono::nanoseconds{date});
}