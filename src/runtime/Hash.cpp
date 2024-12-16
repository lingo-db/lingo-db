#include "lingodb/runtime/helpers.h"
#include "xxhash.h"

EXPORT uint64_t hashVarLenData(lingodb::runtime::VarLen32 str) {
   xxh::hash_t<64> hash = xxh::xxhash<64>(str.getPtr(), str.getLen());
   return hash;
}