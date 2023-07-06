#include "execution/Backend.h"
#include "runtime/helpers.h"
extern "C" runtime::VarLen32 createVarLen32(uint8_t* ptr, uint32_t len); //NOLINT(clang-diagnostic-return-type-c-linkage)
extern "C" uint64_t hashVarLenData(runtime::VarLen32 str);

void execution::visitBareFunctions(const std::function<void(std::string, void*)>& fn) {
   fn("createVarLen32", reinterpret_cast<void*>(&createVarLen32));
   fn("hashVarLenData", reinterpret_cast<void*>(&hashVarLenData));
}
