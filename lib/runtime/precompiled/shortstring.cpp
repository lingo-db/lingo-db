#include "runtime/helpers.h"

EXPORT INLINE __int128 rt_varlen_from_ptr(uint8_t* ptr, uint32_t len) {
   auto x= runtime::VarLen32(ptr,len);
   return *(reinterpret_cast<__int128*>(&x));
}

EXPORT INLINE runtime::Str rt_varlen_to_ref(__int128* varlen) {
   auto obj = reinterpret_cast<runtime::VarLen32*>(varlen);
   return runtime::Str(obj->data(),obj->getLen());
}