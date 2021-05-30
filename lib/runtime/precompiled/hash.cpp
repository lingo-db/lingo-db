#include "runtime/helpers.h"
uint64_t hash64(uint64_t key) {
   //murmur hash 3 mixer
   key ^= key >> 33;
   key *= 0xff51afd7ed558ccd;
   key ^= key >> 33;
   key *= 0xc4ceb9fe1a85ec53;
   key ^= key >> 33;
   return key;
}
extern "C" {
uint64_t _mlir_ciface_hash_int_128(__int128 val) {
   return hash64(val) ^ hash64(val >> 64);
}
uint64_t _mlir_ciface_hash_int_64(uint64_t val) {
   return hash64(val);
}
uint64_t _mlir_ciface_hash_int_32(uint32_t val) {
   return hash64(val);
}
uint64_t _mlir_ciface_hash_int_16(uint16_t val) {
   return hash64(val);
}
uint64_t _mlir_ciface_hash_int_8(uint8_t val) {
   return hash64(val);
}
uint64_t _mlir_ciface_hash_bool(bool val) {
   return hash64(val);
}
uint64_t _mlir_ciface_hash_float_32(float val) {
   return hash64(*(uint32_t*) &val);
}
uint64_t _mlir_ciface_hash_float_64(double val) {
   return hash64(*(uint64_t*) &val);
}

uint64_t _mlir_ciface_hash_binary(runtime::String* val) {
   uint64_t hash = 0;
   size_t i = 0;
   for (; i < val->len() / 8; i++) {
      hash ^= hash64(*((uint64_t*) &val->data()[i]));
   }
   if (i < val->len()) {
      uint64_t remaining = 0;
      for (; i < val->len(); i++) {
         remaining <<= 8;
         uint64_t extended = val->data()[i];
         remaining |= extended;
      }
      hash ^= hash64(remaining);
   }
   return hash;
}
}