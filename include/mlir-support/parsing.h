#ifndef MLIR_SUPPORT_PARSING_H
#define MLIR_SUPPORT_PARSING_H

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <arrow/type_fwd.h>
namespace support {
enum TimeUnit {
   SECOND,
   MILLI,
   MICRO,
   NANO
};
std::pair<uint64_t, uint64_t> getDecimalScaleMultiplier(int32_t scale);
std::pair<uint64_t, uint64_t> parseDecimal(std::string str, int32_t scale);
std::variant<int64_t, double, std::string> parse(std::variant<int64_t, double, std::string> val, arrow::Type::type type, uint32_t param1 = 0, uint32_t param2 = 0);

} // end namespace support

#endif // MLIR_SUPPORT_PARSING_H
