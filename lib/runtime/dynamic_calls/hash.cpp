#include "runtime/xxhash.h"
#include "runtime/helpers.h"

uint64_t rt_hash_long(uint8_t* ptr, uint32_t len){
   xxh::hash_t<64> hash = xxh::xxhash<64>(ptr,len);
   return hash;
}


