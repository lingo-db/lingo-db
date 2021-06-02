#ifndef MLIR_SUPPORT_MLIR_SUPPORT_H
#define MLIR_SUPPORT_MLIR_SUPPORT_H

#include <cstdint>
#include <string>
namespace support{
enum TimeUnit{
   SECOND,
   MILLI,
   MICRO,
   NANO
};
int32_t parseDate32(std::string str);
int64_t parseTimestamp(std::string str, TimeUnit unit);
std::pair<uint64_t, uint64_t> getDecimalScaleMultiplier(unsigned scale);
std::pair<uint64_t, uint64_t> parseDecimal(std::string str, unsigned scale);

} // end namespace support

#endif // MLIR_SUPPORT_MLIR_SUPPORT_H
