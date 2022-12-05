
#ifndef MLIR_SUPPORT_TOSTRING_H
#define MLIR_SUPPORT_TOSTRING_H

#include <arrow/type_fwd.h>
namespace support {
std::string decimalToString(uint64_t low, uint64_t high, int32_t scale);
std::string dateToString(uint64_t);
} // namespace support
#endif //MLIR_SUPPORT_TOSTRING_H
