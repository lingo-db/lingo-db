#include "xxhash.h"
#include "runtime/helpers.h"

EXPORT uint64_t rt_hash_varlen( runtime::VarLen32 str) {
   xxh::hash_t<64> hash = xxh::xxhash<64>(str.getPtr(),str.getLen());
   return hash;
}