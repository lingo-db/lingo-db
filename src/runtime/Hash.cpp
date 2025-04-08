#include "lingodb/runtime/helpers.h"
#include "llvm/Support/xxhash.h"

EXPORT uint64_t hashVarLenData(lingodb::runtime::VarLen32 str) {
   llvm::ArrayRef<uint8_t> data(str.getPtr(), str.getLen());
   return llvm::xxHash64(data);
}
