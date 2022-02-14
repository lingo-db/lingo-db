#include "arrow/util/formatting.h"
#include "arrow/util/value_parsing.h"
#include "runtime/helpers.h"

#include <arrow/type.h>
#include <arrow/util/decimal.h>

//taken from NoisePage
// src: https://github.com/cmu-db/noisepage/blob/c2635d3360dd24a9f7a094b4b8bcd131d99f2d4b/src/execution/sql/operators/like_operators.cpp
// (MIT License, Copyright (c) 2018 CMU Database Group)
#define NextByte(p, plen) ((p)++, (plen)--)
bool like(const char* str, size_t str_len, const char* pattern, size_t pattern_len, char escape) {
   const char *s = str, *p = pattern;
   std::size_t slen = str_len, plen = pattern_len;

   for (; plen > 0 && slen > 0; NextByte(p, plen)) {
      if (*p == escape) {
         // Next pattern character must match exactly, whatever it is
         NextByte(p, plen);

         if (plen == 0 || *p != *s) {
            return false;
         }

         NextByte(s, slen);
      } else if (*p == '%') {
         // Any sequence of '%' wildcards can essentially be replaced by one '%'. Similarly, any
         // sequence of N '_'s will blindly consume N characters from the input string. Process the
         // pattern until we reach a non-wildcard character.
         NextByte(p, plen);
         while (plen > 0) {
            if (*p == '%') {
               NextByte(p, plen);
            } else if (*p == '_') {
               if (slen == 0) {
                  return false;
               }
               NextByte(s, slen);
               NextByte(p, plen);
            } else {
               break;
            }
         }

         // If we've reached the end of the pattern, the tail of the input string is accepted.
         if (plen == 0) {
            return true;
         }

         if (*p == escape) {
            NextByte(p, plen);
            if (plen == 0) {
               return false;
            }
         }

         while (slen > 0) {
            if (like(s, slen, p, plen, escape)) {
               return true;
            }
            NextByte(s, slen);
         }
         // No match
         return false;
      } else if (*p == '_') {
         // '_' wildcard matches a single character in the input
         NextByte(s, slen);
      } else if (*p == *s) {
         // Exact character match
         NextByte(s, slen);
      } else {
         // Unmatched!
         return false;
      }
   }
   while (plen > 0 && *p == '%') {
      NextByte(p, plen);
   }
   return slen == 0 && plen == 0;
}
//end taken from noisepage

extern "C" bool rt_cmp_string_like(bool null, runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (null) {
      return false;
   } else {
      return like((str1).data(), (str1).getLen(), (str2).data(), (str2).getLen(), '\\');
   }
}
extern "C" bool rt_cmp_string_ends_with(bool null, runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (null) {
      return false;
   } else {
      if (str1.getLen() < str2.getLen()) return false;
      return memcmp(str1.data() + str1.getLen() - str2.getLen(), str2.data(), str2.getLen()) == 0;
   }
}
extern "C" bool rt_cmp_string_starts_with(bool null, runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (null) {
      return false;
   } else {
      if (str1.getLen() < str2.getLen()) return false;
      return memcmp(str1.data(), str2.data(), str2.getLen()) == 0;
   }
}

//taken from gandiva
//source https://github.com/apache/arrow/blob/41d115071587d68891b219cc137551d3ea9a568b/cpp/src/gandiva/gdv_function_stubs.cc
//Apache-2.0 License
#define CAST_NUMERIC_FROM_STRING(OUT_TYPE, ARROW_TYPE, TYPE_NAME)                                                                          \
   extern "C" OUT_TYPE rt_cast_string_##TYPE_NAME(bool null, runtime::VarLen32 str) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      if (null) return (OUT_TYPE) 0;                                                                                                       \
      char* data = (str).data();                                                                                                           \
      int32_t len = (str).getLen();                                                                                                        \
      OUT_TYPE val = 0;                                                                                                                    \
      /* trim leading and trailing spaces */                                                                                               \
      int32_t trimmed_len;                                                                                                                 \
      int32_t start = 0, end = len - 1;                                                                                                    \
      while (start <= end && data[start] == ' ') {                                                                                         \
         ++start;                                                                                                                          \
      }                                                                                                                                    \
      while (end >= start && data[end] == ' ') {                                                                                           \
         --end;                                                                                                                            \
      }                                                                                                                                    \
      trimmed_len = end - start + 1;                                                                                                       \
      const char* trimmed_data = data + start;                                                                                             \
      if (!arrow::internal::ParseValue<ARROW_TYPE>(trimmed_data, trimmed_len, &val)) {                                                     \
         std::string err =                                                                                                                 \
            "Failed to cast the string " + std::string(data, len) + " to " #OUT_TYPE;                                                      \
         /*gdv_fn_context_set_error_msg(context, err.c_str());*/                                                                           \
      }                                                                                                                                    \
      return val;                                                                                                                          \
   }

CAST_NUMERIC_FROM_STRING(int64_t, arrow::Int64Type, int)
CAST_NUMERIC_FROM_STRING(float, arrow::FloatType, float32)
CAST_NUMERIC_FROM_STRING(double, arrow::DoubleType, float64)
//end taken from gandiva

extern "C" __int128 rt_cast_string_decimal(bool null, runtime::VarLen32 string, unsigned reqScale) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return 0;
   }
   int32_t precision;
   int32_t scale;
   arrow::Decimal128 decimalrep;
   if (!arrow::Decimal128::FromString(string.str(), &decimalrep, &precision, &scale).ok()) {
      //todo
   }
   auto x = decimalrep.Rescale(scale, reqScale);
   decimalrep = x.ValueUnsafe();
   __int128 res = decimalrep.high_bits();
   res <<= 64;
   res |= decimalrep.low_bits();
   return res;
}
#define CAST_NUMERIC_TO_STRING(IN_TYPE, ARROW_TYPE, TYPE_NAME)                                                                                \
   extern "C" runtime::VarLen32 rt_cast_##TYPE_NAME##_string(bool null, IN_TYPE value) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      if (null) {                                                                                                                             \
         return runtime::VarLen32(nullptr, 0);                                                                                                \
      }                                                                                                                                       \
      arrow::internal::StringFormatter<ARROW_TYPE> formatter;                                                                                 \
      uint8_t* data = nullptr;                                                                                                                \
      size_t len = 0;                                                                                                                         \
      arrow::Status status = formatter(value, [&](arrow::util::string_view v) {                                                               \
         len = v.length();                                                                                                                    \
         data = new uint8_t[len];                                                                                                             \
         memcpy(data, v.data(), len);                                                                                                         \
         return arrow::Status::OK();                                                                                                          \
      });                                                                                                                                     \
      return runtime::VarLen32(data, len);                                                                                                    \
   }

CAST_NUMERIC_TO_STRING(int64_t, arrow::Int64Type, int)
CAST_NUMERIC_TO_STRING(float, arrow::FloatType, float32)
CAST_NUMERIC_TO_STRING(double, arrow::DoubleType, float64)

extern "C" runtime::VarLen32 rt_cast_decimal_string(bool null, uint64_t low, uint64_t high, uint32_t scale) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return runtime::VarLen32(nullptr, 0);
   }
   arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
   std::string str = decimalrep.ToString(scale);
   size_t len = str.length();
   uint8_t* data = new uint8_t[len];
   memcpy(data, str.data(), len);

   return runtime::VarLen32(data, len);
}

EXPORT runtime::VarLen32 rt_cast_char_string(bool null, uint64_t val, size_t bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   char* data = new char[bytes];
   memcpy(data, &val, bytes);
   return runtime::VarLen32((uint8_t*) data, bytes);
}
//taken from apache gandiva
//source: https://github.com/apache/arrow/blob/master/cpp/src/gandiva/precompiled/string_ops.cc
//Apache-2.0 License
__attribute__((always_inline)) inline int64_t mem_compare(const char* left, int64_t left_len, const char* right,
                                                          int64_t right_len) {
   int min = left_len;
   if (right_len < min) {
      min = right_len;
   }

   int cmp_ret = memcmp(left, right, min);
   if (cmp_ret != 0) {
      return cmp_ret;
   } else {
      return left_len - right_len;
   }
}
//end taken from apache gandiva
#define STR_CMP(NAME, OP)                                                                            \
   extern "C" bool rt_cmp_string_##NAME(bool null, runtime::VarLen32 str1, runtime::VarLen32 str2) { \
      if (null) {                                                                                    \
         return false;                                                                               \
      } else {                                                                                       \
         return mem_compare((str1).data(), (str1).getLen(), (str2).data(), (str2).getLen()) OP 0;    \
      }                                                                                              \
   }

STR_CMP(lt, <)
STR_CMP(lte, <=)
STR_CMP(gt, >)
STR_CMP(gte, >=)

extern "C" void rt_cpy(runtime::Bytes to, runtime::Bytes from) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   memcpy(to.getPtr(), from.getPtr(), from.getSize());
}
extern "C" void rt_fill(runtime::Bytes from, char val) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   memset(from.getPtr(), val, from.getSize());
}
struct P64 {
   uint64_t a;
   uint64_t b;
};

uint64_t rt_hash_long(uint8_t* ptr, uint32_t len);
EXPORT uint64_t rt_hash_varlen(P64 str) {
   auto* obj = reinterpret_cast<runtime::VarLen32*>(&str);
   return rt_hash_long(obj->getPtr(), obj->getLen());
}
EXPORT bool rt_cmp_string_eq(bool null, runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (null) return false;

   if (str1.getLen() != str2.getLen()) return false;
   return memcmp(str1.data(), str2.data(), str1.getLen()) == 0;
}
EXPORT bool rt_cmp_string_neq(bool null, runtime::VarLen32 str1, runtime::VarLen32 str2) {
   if (null) return false;

   if (str1.getLen() != str2.getLen()) return true;
   return memcmp(str1.data(), str2.data(), str1.getLen()) != 0;
}
EXPORT P64 rt_varlen_from_ptr(uint8_t* ptr, uint32_t len) {
   auto x = runtime::VarLen32(ptr, len);
   return *(reinterpret_cast<P64*>(&x));
}

EXPORT runtime::Bytes rt_varlen_to_ref(P64* varlen) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto* obj = reinterpret_cast<runtime::VarLen32*>(varlen);
   return runtime::Bytes((uint8_t*) obj->data(), obj->getLen());
}

struct Vec {
   size_t len;
   size_t cap;
   runtime::Bytes bytes;
   Vec(size_t len, size_t cap, size_t numBytes) : len(len), cap(cap), bytes((uint8_t*) malloc(numBytes), numBytes) {}
};
struct AggrHt {
   struct Entry {
      size_t next;
      size_t hashValue;
      //kv follows
   };
   size_t numValues;
   size_t capacity;
   runtime::Bytes values;
   runtime::Bytes ht;
   //initial value follows...
   AggrHt(size_t initialCapacity, size_t typeSize) : numValues(0), capacity(initialCapacity), values((uint8_t*) malloc(initialCapacity * typeSize), initialCapacity * typeSize), ht((uint8_t*) malloc(initialCapacity * 2 * sizeof(uint64_t)), initialCapacity * 2 * sizeof(uint64_t)) {
      ht.fill(0xff);
   }
};
EXPORT Vec* rt_create_vec(size_t sizeOfType, size_t initialCapacity) {
   return new Vec(0, initialCapacity, initialCapacity * sizeOfType);
}
EXPORT void rt_resize_vec(Vec* v) {
   v->cap *= 2;
   v->bytes.resize(2);
}
EXPORT AggrHt* rt_create_aggr_ht(size_t typeSize, size_t initialCapacity) {
   return new (malloc(sizeof(AggrHt) + typeSize)) AggrHt(initialCapacity, typeSize);
}
EXPORT void rt_resize_aggr_ht(AggrHt* aggrHt, size_t typeSize) {
   size_t old = aggrHt->values.getSize();
   aggrHt->values.resize(2);
   aggrHt->ht.resize(2);
   aggrHt->ht.fill(0xff);
   aggrHt->capacity *= 2;
   size_t* ht = (size_t*) aggrHt->ht.getPtr();
   size_t hashMask = (aggrHt->ht.getSize() / sizeof(size_t)) - 1;
   auto valuesPtr = aggrHt->values.getPtr();
   assert((old / typeSize) == aggrHt->numValues);
   for (size_t i = 0; i < aggrHt->numValues; i++) {
      //assert((i * (sizeof(AggrHt::Entry) + typeSize)) < old);
      AggrHt::Entry* entry = (AggrHt::Entry*) &valuesPtr[i * typeSize];
      auto pos = entry->hashValue & hashMask;
      auto previousPtr = ht[pos];
      ht[pos] = i;
      entry->next = previousPtr;
   }
}
//!llvm.ptr<struct<(ptr<>, i64, struct<(ptr<i64>, i64)>, i64)>>
struct JoinHt {
   struct Entry {
      Entry* next;
      //kv follows
   };
   runtime::Bytes values;
   size_t numValues;
   runtime::Bytes ht;
   size_t htMask;
   JoinHt(const runtime::Bytes& values, size_t numValues, size_t htMask, size_t htSize) : values(values), numValues(numValues), ht((uint8_t*) malloc(htSize * sizeof(uint64_t)), htSize * sizeof(uint64_t)), htMask(htMask) {}
};
EXPORT uint64_t next_pow_2(uint64_t v) {
   v--;
   v |= v >> 1;
   v |= v >> 2;
   v |= v >> 4;
   v |= v >> 8;
   v |= v >> 16;
   v |= v >> 32;
   v++;
   return v;
}

EXPORT JoinHt* rt_build_join_ht(Vec* v, size_t typeSize) {
   size_t htSize = next_pow_2(v->len);
   size_t htMask = htSize - 1;
   auto joinHt = new JoinHt(v->bytes, v->len, htMask, htSize);
   joinHt->ht.fill(0x00);
   auto valuesPtr = joinHt->values.getPtr();
   JoinHt::Entry** ht = (JoinHt::Entry**) joinHt->ht.getPtr();
   for (size_t i = 0; i < v->len; i++) {
      auto entry = (JoinHt::Entry*) &valuesPtr[i * typeSize];
      auto pos = ((size_t) entry->next) & htMask;
      auto previousPtr = ht[pos];
      ht[pos] = entry;
      entry->next = previousPtr;
   }
   return joinHt;
}