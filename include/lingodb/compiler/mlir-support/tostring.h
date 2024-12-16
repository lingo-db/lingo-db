#ifndef LINGODB_COMPILER_MLIR_SUPPORT_TOSTRING_H
#define LINGODB_COMPILER_MLIR_SUPPORT_TOSTRING_H

#include <arrow/type_fwd.h>
namespace lingodb::compiler::support {
std::string decimalToString(uint64_t low, uint64_t high, int32_t scale);
std::string dateToString(uint64_t);
} // namespace lingodb::compiler::support
#endif //LINGODB_COMPILER_MLIR_SUPPORT_TOSTRING_H
