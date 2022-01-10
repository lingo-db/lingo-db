#include "runtime/helpers.h"

struct P64{
   uint64_t a;
   uint64_t b;
};
EXPORT INLINE P64 rt_varlen_from_ptr(uint8_t* ptr, uint32_t len) {
   auto x = runtime::VarLen32(ptr, len);
   return *(reinterpret_cast<P64*>(&x));
}

EXPORT INLINE runtime::Str rt_varlen_to_ref(P64* varlen) {
   auto obj = reinterpret_cast<runtime::VarLen32*>(varlen);
   return runtime::Str(obj->data(), obj->getLen());
}

static uint64_t hash64(uint64_t val) {
   auto m= val * 0xbf58476d1ce4e5b9;
   return m;
}
EXPORT INLINE uint64_t rt_hash_64(uint64_t val) {
   return hash64(val);
}
EXPORT INLINE uint64_t rt_hash_combine(uint64_t h1, uint64_t h2) {
      // Based on Hash128to64() from cityhash.xxh3
      static constexpr auto k_mul = uint64_t(0x9ddfea08eb382d69);
      uint64_t a = (h1 ^ h2) * k_mul;
      a ^= (a >> 47u);
      uint64_t b = (h2 ^ a) * k_mul;
      b ^= (b >> 47u);
      b *= k_mul;
      return b;
}
uint64_t rt_hash_long(uint8_t* ptr, uint32_t len);
EXPORT INLINE uint64_t rt_hash_varlen(P64 str) {
   auto obj = reinterpret_cast<runtime::VarLen32*>(&str);
   if (obj->getLen() <= runtime::VarLen32::SHORT_LEN) {
      return rt_hash_combine(hash64(str.a), hash64(str.b));
   } else {
      return rt_hash_long(obj->getPtr(),obj->getLen());
   }
}
EXPORT INLINE bool rt_cmp_string_eq(bool null, runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (null) return false;

   if (str1.getLen() != str2.getLen()) return false;

   if (str1.first4 != str2.first4) {
      //assert(memcmp(str1.data(), str2.data(), str1.getLen()) != 0);
      return false;
   }
   if (str1.getLen() <= runtime::VarLen32::SHORT_LEN) {
      return str1.last8 == str2.last8;
   }
   return memcmp(str1.data(), str2.data(), str1.getLen()) == 0;
}
