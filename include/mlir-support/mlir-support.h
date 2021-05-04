#ifndef MLIR_SUPPORT_MLIR_SUPPORT_H
#define MLIR_SUPPORT_MLIR_SUPPORT_H

#include "arrow/type_fwd.h"
#include <cstdint>
#include <string>
namespace support {
int32_t parseDate32(std::string str);
int64_t parseTimestamp(std::string str, arrow::TimeUnit::type unit);
std::pair<uint64_t, uint64_t> getDecimalScaleMultiplier(unsigned scale);
std::pair<uint64_t, uint64_t> parseDecimal(std::string str, unsigned scale);

} // end namespace support

#endif // MLIR_SUPPORT_MLIR_SUPPORT_H
