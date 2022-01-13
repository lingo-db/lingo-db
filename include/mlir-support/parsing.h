#ifndef MLIR_SUPPORT_MLIR_SUPPORT_H
#define MLIR_SUPPORT_MLIR_SUPPORT_H

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include <arrow/type_fwd.h>
namespace support{
enum TimeUnit{
   SECOND,
   MILLI,
   MICRO,
   NANO
};
std::pair<uint64_t, uint64_t> getDecimalScaleMultiplier(unsigned scale);
std::pair<uint64_t, uint64_t> parseDecimal(std::string str, unsigned scale);
std::vector<std::byte> parse(std::variant<int64_t,double,std::string> val,arrow::Type::type type,uint32_t param1=0, uint32_t param2=0);

} // end namespace support

#endif // MLIR_SUPPORT_MLIR_SUPPORT_H
