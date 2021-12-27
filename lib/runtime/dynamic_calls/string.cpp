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

extern "C" bool rt_cmp_string_like(bool null, runtime::Str str1, runtime::Str str2) {
   if (null) {
      return false;
   } else {
      return like((str1).data(), (str1).len(), (str2).data(), (str2).len(), '\\');
   }
}

//taken from gandiva
//source https://github.com/apache/arrow/blob/41d115071587d68891b219cc137551d3ea9a568b/cpp/src/gandiva/gdv_function_stubs.cc
//Apache-2.0 License
#define CAST_NUMERIC_FROM_STRING(OUT_TYPE, ARROW_TYPE, TYPE_NAME)                                                                     \
   extern "C" OUT_TYPE rt_cast_string_##TYPE_NAME(bool null, runtime::Str str) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      if (null) return (OUT_TYPE) 0;                                                                                                  \
      char* data = (str).data();                                                                                                      \
      int32_t len = (str).len();                                                                                                      \
      OUT_TYPE val = 0;                                                                                                               \
      /* trim leading and trailing spaces */                                                                                          \
      int32_t trimmed_len;                                                                                                            \
      int32_t start = 0, end = len - 1;                                                                                               \
      while (start <= end && data[start] == ' ') {                                                                                    \
         ++start;                                                                                                                     \
      }                                                                                                                               \
      while (end >= start && data[end] == ' ') {                                                                                      \
         --end;                                                                                                                       \
      }                                                                                                                               \
      trimmed_len = end - start + 1;                                                                                                  \
      const char* trimmed_data = data + start;                                                                                        \
      if (!arrow::internal::ParseValue<ARROW_TYPE>(trimmed_data, trimmed_len, &val)) {                                                \
         std::string err =                                                                                                            \
            "Failed to cast the string " + std::string(data, len) + " to " #OUT_TYPE;                                                 \
         /*gdv_fn_context_set_error_msg(context, err.c_str());*/                                                                      \
      }                                                                                                                               \
      return val;                                                                                                                     \
   }

CAST_NUMERIC_FROM_STRING(int64_t, arrow::Int64Type, int)
CAST_NUMERIC_FROM_STRING(float, arrow::FloatType, float32)
CAST_NUMERIC_FROM_STRING(double, arrow::DoubleType, float64)
//end taken from gandiva

extern "C" __int128 rt_cast_string_decimal(bool null, runtime::Str string, unsigned reqScale) { // NOLINT (clang-diagnostic-return-type-c-linkage)
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
#define CAST_NUMERIC_TO_STRING(IN_TYPE, ARROW_TYPE, TYPE_NAME)                                                                           \
   extern "C" runtime::Str rt_cast_##TYPE_NAME##_string(bool null, IN_TYPE value) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      if (null) {                                                                                                                        \
         return runtime::Str(nullptr, 0);                                                                                                \
      }                                                                                                                                  \
      arrow::internal::StringFormatter<ARROW_TYPE> formatter;                                                                            \
      char* data = nullptr;                                                                                                              \
      size_t len = 0;                                                                                                                    \
      arrow::Status status = formatter(value, [&](arrow::util::string_view v) {                                                          \
         len = v.length();                                                                                                               \
         data = new char[len];                                                                                                           \
         memcpy(data, v.data(), len);                                                                                                    \
         return arrow::Status::OK();                                                                                                     \
      });                                                                                                                                \
      return runtime::Str(data, len);                                                                                                    \
   }

CAST_NUMERIC_TO_STRING(int64_t, arrow::Int64Type, int)
CAST_NUMERIC_TO_STRING(float, arrow::FloatType, float32)
CAST_NUMERIC_TO_STRING(double, arrow::DoubleType, float64)

extern "C" runtime::Str rt_cast_decimal_string(bool null, uint64_t low, uint64_t high, uint32_t scale) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return runtime::Str(nullptr, 0);
   }
   arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
   std::string str = decimalrep.ToString(scale);
   size_t len = str.length();
   char* data = new char[len];
   memcpy(data, str.data(), len);

   return runtime::Str(data, len);
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

#define STR_CMP(NAME, OP)                                                                   \
   extern "C" bool rt_cmp_string_##NAME(bool null, runtime::Str str1, runtime::Str str2) {  \
      if (null) {                                                                           \
         return false;                                                                      \
      } else {                                                                              \
         return mem_compare((str1).data(), (str1).len(), (str2).data(), (str2).len()) OP 0; \
      }                                                                                     \
   }

STR_CMP(eq, ==)
STR_CMP(neq, !=)
STR_CMP(lt, <)
STR_CMP(lte, <=)
STR_CMP(gt, >)
STR_CMP(gte, >=)

extern "C" void rt_cpy(runtime::Str to, runtime::Str from) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   memcpy(to.data(), from.data(), from.len());
}
extern "C" void rt_fill(runtime::Str from, char val) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   memset(from.data(), val, from.len());
}