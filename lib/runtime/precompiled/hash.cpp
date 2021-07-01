#include "runtime/helpers.h"
#include <immintrin.h>

extern "C" {
uint64_t _mlir_ciface_hash_int_128(uint64_t h, __int128 val) {
   return _mm_crc32_u64(_mm_crc32_u64(h, val >> 64), val);
}
uint64_t _mlir_ciface_hash_int_64(uint64_t h, uint64_t val) {
   return _mm_crc32_u64(h, val);
}
uint64_t _mlir_ciface_hash_int_32(uint64_t h, uint32_t val) {
   return _mm_crc32_u64(h, val);
}
uint64_t _mlir_ciface_hash_int_16(uint64_t h, uint16_t val) {
   return _mm_crc32_u64(h, val);
}
uint64_t _mlir_ciface_hash_int_8(uint64_t h, uint8_t val) {
   return _mm_crc32_u64(h, val);
}
uint64_t _mlir_ciface_hash_bool(uint64_t h, bool val) {
   return _mm_crc32_u64(h, val);
}
uint64_t _mlir_ciface_hash_float_32(uint64_t h, float val) {
   return _mm_crc32_u64(h, *(uint32_t*) &val);
}
uint64_t _mlir_ciface_hash_float_64(uint64_t h, double val) {
   return _mm_crc32_u64(h, *(uint64_t*) &val);
}

uint64_t _mlir_ciface_hash_binary(uint64_t hash, runtime::String* val) {
   size_t i = 0;
   for (; i < val->len() / 8; i++) {
      hash = _mm_crc32_u64(hash,*((uint64_t*) &val->data()[i]));
   }
   if (i < val->len()) {
      uint64_t remaining = 0;
      for (; i < val->len(); i++) {
         remaining <<= 8;
         uint64_t extended = val->data()[i];
         remaining |= extended;
      }
      hash = _mm_crc32_u64(hash,remaining);
   }
   return hash;
}
}