#include "arrow/util/formatting.h"
#include "arrow/util/value_parsing.h"
#include "runtime/helpers.h"
#include <arrow/type.h>
#include <arrow/util/decimal.h>

#define NextByte(p, plen) ((p)++, (plen)--)

//taken from noisepage
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
extern "C" bool _mlir_ciface_cmp_string_like(bool null, runtime::Str str1, runtime::Str str2) {
   if (null) {
      return false;
   } else {
      return like((str1).data(), (str1).len(), (str2).data(), (str2).len(), '\\');
   }
}

//taken from gandiva

#define CAST_NUMERIC_FROM_STRING(OUT_TYPE, ARROW_TYPE, TYPE_NAME)                                                                                   \
   extern "C" OUT_TYPE _mlir_ciface_cast_string_##TYPE_NAME(bool null, runtime::Str str) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      if (null) return (OUT_TYPE) 0;                                                                                                                \
      char* data = (str).data();                                                                                                                   \
      int32_t len = (str).len();                                                                                                                   \
      OUT_TYPE val = 0;                                                                                                                             \
      /* trim leading and trailing spaces */                                                                                                        \
      int32_t trimmed_len;                                                                                                                          \
      int32_t start = 0, end = len - 1;                                                                                                             \
      while (start <= end && data[start] == ' ') {                                                                                                  \
         ++start;                                                                                                                                   \
      }                                                                                                                                             \
      while (end >= start && data[end] == ' ') {                                                                                                    \
         --end;                                                                                                                                     \
      }                                                                                                                                             \
      trimmed_len = end - start + 1;                                                                                                                \
      const char* trimmed_data = data + start;                                                                                                      \
      if (!arrow::internal::ParseValue<ARROW_TYPE>(trimmed_data, trimmed_len, &val)) {                                                              \
         std::string err =                                                                                                                          \
            "Failed to cast the string " + std::string(data, len) + " to " #OUT_TYPE;                                                               \
         /*gdv_fn_context_set_error_msg(context, err.c_str());*/                                                                                    \
      }                                                                                                                                             \
      return val;                                                                                                                                   \
   }

CAST_NUMERIC_FROM_STRING(int64_t, arrow::Int64Type, int)
CAST_NUMERIC_FROM_STRING(float, arrow::FloatType, float32)
CAST_NUMERIC_FROM_STRING(double, arrow::DoubleType, float64)

extern "C" __int128 _mlir_ciface_cast_string_decimal(bool null, runtime::Str string, unsigned reqScale) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return 0;
   }
   int32_t precision;
   int32_t scale;
   arrow::Decimal128 decimalrep;
   if (arrow::Decimal128::FromString(string.str(), &decimalrep, &precision, &scale) != arrow::Status::OK()) {
      //todo
   }
   auto x = decimalrep.Rescale(scale, reqScale);
   decimalrep = x.ValueUnsafe();
   __int128 res = decimalrep.high_bits();
   res <<= 64;
   res |= decimalrep.low_bits();
   return res;
}
#define CAST_NUMERIC_TO_STRING(IN_TYPE, ARROW_TYPE, TYPE_NAME)                                                                                        \
   extern "C" runtime::Str _mlir_ciface_cast_##TYPE_NAME##_string(bool null, IN_TYPE value) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      if (null) {                                                                                                                                     \
         return runtime::Str(nullptr, 0);                                                                                                          \
      }                                                                                                                                               \
      arrow::internal::StringFormatter<ARROW_TYPE> formatter;                                                                                         \
      char* data = nullptr;                                                                                                                           \
      size_t len = 0;                                                                                                                                 \
      arrow::Status status = formatter(value, [&](arrow::util::string_view v) {                                                                       \
         len = v.length();                                                                                                                            \
         data = new char[len];                                                                                                                        \
         memcpy(data, v.data(), len);                                                                                                                 \
         return arrow::Status::OK();                                                                                                                  \
      });                                                                                                                                             \
      return runtime::Str(data, len);                                                                                                              \
   }

CAST_NUMERIC_TO_STRING(int64_t, arrow::Int64Type, int)
CAST_NUMERIC_TO_STRING(float, arrow::FloatType, float32)
CAST_NUMERIC_TO_STRING(double, arrow::DoubleType, float64)

extern "C" runtime::Str _mlir_ciface_cast_decimal_string(bool null, uint64_t low, uint64_t high, uint32_t scale) {// NOLINT (clang-diagnostic-return-type-c-linkage)
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
/*extern "C" bool _mlir_ciface_cmp_string_eq(bool null, runtime::Str str1, runtime::Str str2) {
if (null) {
return false;
} else {
return str1.str()==str2.str();
}
}*/

