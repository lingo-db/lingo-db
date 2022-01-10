#include "runtime/helpers.h"

struct P64{
   uint64_t a;
   uint64_t b;
};
EXPORT INLINE P64 rt_varlen_from_ptr(uint8_t* ptr, uint32_t len) {
   auto x= runtime::VarLen32(ptr,len);
   return *(reinterpret_cast<P64*>(&x));
}

EXPORT INLINE runtime::Str rt_varlen_to_ref(P64* varlen) {
   auto obj = reinterpret_cast<runtime::VarLen32*>(varlen);
   return runtime::Str(obj->data(),obj->getLen());
}