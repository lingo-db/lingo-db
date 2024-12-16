#include "lingodb/runtime/StringRuntime.h"
#include "arrow/util/formatting.h"
#include "arrow/util/value_parsing.h"
#include "lingodb/runtime/helpers.h"

#include <arrow/type.h>
#include <arrow/util/decimal.h>

//taken from NoisePage
// src: https://github.com/cmu-db/noisepage/blob/c2635d3360dd24a9f7a094b4b8bcd131d99f2d4b/src/execution/sql/operators/like_operators.cpp
// (MIT License, Copyright (c) 2018 CMU Database Group)
#define NextByte(p, plen) ((p)++, (plen)--)
namespace {
bool iterativeLike(const char* str, size_t strLen, const char* pattern, size_t patternLen, char escape) {
   const char *s = str, *p = pattern;
   std::size_t slen = strLen, plen = patternLen;

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
            if (iterativeLike(s, slen, p, plen, escape)) {
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
} // namespace
//end taken from noisepage

bool lingodb::runtime::StringRuntime::like(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   return iterativeLike((str1).data(), (str1).getLen(), (str2).data(), (str2).getLen(), '\\');
}
bool lingodb::runtime::StringRuntime::endsWith(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   if (str1.getLen() < str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()).ends_with(std::string_view(str2.data(), str2.getLen()));
}
bool lingodb::runtime::StringRuntime::startsWith(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   if (str1.getLen() < str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()).starts_with(std::string_view(str2.data(), str2.getLen()));
}

//taken from gandiva
//source https://github.com/apache/arrow/blob/41d115071587d68891b219cc137551d3ea9a568b/cpp/src/gandiva/gdv_function_stubs.cc
//Apache-2.0 License
#define CAST_NUMERIC_FROM_STRING(OUT_TYPE, ARROW_TYPE, TYPE_NAME)                                                               \
   OUT_TYPE lingodb::runtime::StringRuntime::to##TYPE_NAME(lingodb::runtime::VarLen32 str) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      char* data = (str).data();                                                                                                \
      int32_t len = (str).getLen();                                                                                             \
      OUT_TYPE val = 0;                                                                                                         \
      /* trim leading and trailing spaces */                                                                                    \
      int32_t trimmed_len;                                                                                                      \
      int32_t start = 0, end = len - 1;                                                                                         \
      while (start <= end && data[start] == ' ') {                                                                              \
         ++start;                                                                                                               \
      }                                                                                                                         \
      while (end >= start && data[end] == ' ') {                                                                                \
         --end;                                                                                                                 \
      }                                                                                                                         \
      trimmed_len = end - start + 1;                                                                                            \
      const char* trimmed_data = data + start;                                                                                  \
      if (!arrow::internal::ParseValue<ARROW_TYPE>(trimmed_data, trimmed_len, &val)) {                                          \
         std::string err =                                                                                                      \
            "Failed to cast the string " + std::string(data, len) + " to " #OUT_TYPE;                                           \
         /*gdv_fn_context_set_error_msg(context, err.c_str());*/                                                                \
      }                                                                                                                         \
      return val;                                                                                                               \
   }

CAST_NUMERIC_FROM_STRING(int64_t, arrow::Int64Type, Int)
CAST_NUMERIC_FROM_STRING(float, arrow::FloatType, Float32)
CAST_NUMERIC_FROM_STRING(double, arrow::DoubleType, Float64)
//end taken from gandiva

__int128 lingodb::runtime::StringRuntime::toDecimal(lingodb::runtime::VarLen32 string, int32_t reqScale) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   int32_t precision;
   int32_t scale;
   arrow::Decimal128 decimalrep;
   if (!arrow::Decimal128::FromString(string.str(), &decimalrep, &precision, &scale).ok()) {
      throw std::runtime_error("could not cast decimal");
   }
   auto x = decimalrep.Rescale(scale, reqScale);
   decimalrep = x.ValueUnsafe();
   __int128 res = decimalrep.high_bits();
   res <<= 64;
   res |= decimalrep.low_bits();
   return res;
}
#define CAST_NUMERIC_TO_STRING(IN_TYPE, ARROW_TYPE, TYPE_NAME)                                                                     \
   lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::from##TYPE_NAME(IN_TYPE value) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      arrow::internal::StringFormatter<ARROW_TYPE> formatter;                                                                      \
      uint8_t* data = nullptr;                                                                                                     \
      size_t len = 0;                                                                                                              \
      arrow::Status status = formatter(value, [&](std::string_view v) {                                                    \
         len = v.length();                                                                                                         \
         data = new uint8_t[len];                                                                                                  \
         memcpy(data, v.data(), len);                                                                                              \
         return arrow::Status::OK();                                                                                               \
      });                                                                                                                          \
      return lingodb::runtime::VarLen32(data, len);                                                                                         \
   }

CAST_NUMERIC_TO_STRING(int64_t, arrow::Int64Type, Int)
CAST_NUMERIC_TO_STRING(float, arrow::FloatType, Float32)
CAST_NUMERIC_TO_STRING(double, arrow::DoubleType, Float64)
CAST_NUMERIC_TO_STRING(bool, arrow::BooleanType, Bool)

lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::fromDecimal(__int128 val, int32_t scale) { // NOLINT (clang-diagnostic-return-type-c-linkage)

   arrow::Decimal128 decimalrep(arrow::BasicDecimal128(val >> 64, val));
   std::string str = decimalrep.ToString(scale);
   return lingodb::runtime::VarLen32::fromString(str);
}

lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::fromChar(uint64_t val, size_t bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   char* data = new char[bytes];
   memcpy(data, &val, bytes);
   return lingodb::runtime::VarLen32((uint8_t*) data, bytes);
}

#define STR_CMP(NAME, OP)                                                                                  \
   bool lingodb::runtime::StringRuntime::compare##NAME(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {            \
      return std::string_view(str1.data(), str1.getLen()) OP std::string_view(str2.data(), str2.getLen()); \
   }

STR_CMP(Lt, <)
STR_CMP(Lte, <=)
STR_CMP(Gt, >)
STR_CMP(Gte, >=)

bool lingodb::runtime::StringRuntime::compareEq(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   if (str1.getLen() != str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()) == std::string_view(str2.data(), str2.getLen());
}
bool lingodb::runtime::StringRuntime::compareNEq(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   if (str1.getLen() != str2.getLen()) return true;
   return std::string_view(str1.data(), str1.getLen()) != std::string_view(str2.data(), str2.getLen());
}


EXPORT char* rt_varlen_to_ref(lingodb::runtime::VarLen32* varlen) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return varlen->data();
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::substr(lingodb::runtime::VarLen32 str, size_t from, size_t to) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   from -= 1;
   to = std::min((size_t) str.getLen(), to);
   if (from > to || str.getLen() < to) throw std::runtime_error("can not perform substring operation");
   return lingodb::runtime::VarLen32::fromString(str.str().substr(from, to - from));
}

size_t lingodb::runtime::StringRuntime::findMatch(VarLen32 str, VarLen32 needle, size_t start, size_t end) {
   constexpr size_t invalidPos = 0x8000000000000000;
   if (start >= invalidPos) return invalidPos;
   if (start + needle.getLen() > end) return invalidPos;
   size_t found = std::string_view(str.data(), str.getLen()).find(std::string_view(needle.data(), needle.getLen()), start);

   if (found == std::string::npos || found + needle.getLen() > end) return invalidPos;
   return found + needle.getLen();
}
size_t lingodb::runtime::StringRuntime::findNext(VarLen32 str, VarLen32 needle, size_t start) {
   constexpr size_t invalidPos = 0x8000000000000000;
   if (start >= invalidPos) return invalidPos;
   size_t found = std::string_view(str.data(), str.getLen()).find(std::string_view(needle.data(), needle.getLen()), start);

   if (found == std::string::npos) return invalidPos;
   return found;
}
namespace {
void toUpper(char* str, size_t len) {
   for (auto i = 0ul; i < len; i++) {
      str[i] = std::toupper(str[i]);
   }
}
} // namespace
int64_t lingodb::runtime::StringRuntime::len(VarLen32 str) {
   return str.getLen();
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::toUpper(lingodb::runtime::VarLen32 str) {
   if (str.isShort()) {
      ::toUpper(str.data(), str.getLen());
      return str;
   } else {
      char* copied = new char[str.getLen()];
      memcpy(copied, str.data(), str.getLen());
      ::toUpper(copied, str.getLen());
      return lingodb::runtime::VarLen32((uint8_t*) copied, str.getLen());
   }
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::concat(lingodb::runtime::VarLen32 a, lingodb::runtime::VarLen32 b) {
   auto totalLength = a.getLen() + b.getLen();
   if (totalLength <= lingodb::runtime::VarLen32::shortLen) {
      uint8_t data[lingodb::runtime::VarLen32::shortLen];
      memcpy(data, a.data(), a.getLen());
      memcpy(&data[a.getLen()], b.data(), b.getLen());
      return lingodb::runtime::VarLen32(data, totalLength);
   } else {
      char* copied = new char[totalLength];
      memcpy(copied, a.data(), a.getLen());
      memcpy(&copied[a.getLen()], b.data(), b.getLen());
      return lingodb::runtime::VarLen32((uint8_t*) copied, totalLength);
   }
}
int64_t lingodb::runtime::StringRuntime::toDate(lingodb::runtime::VarLen32 str) {
   int32_t res;
   arrow::internal::ParseValue<arrow::Date32Type>(str.data(), str.getLen(), &res);
   int64_t date64 = static_cast<int64_t>(res) * 24 * 60 * 60 * 1000000000ll;
   return date64;
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::fromDate(int64_t date) {
   static arrow_vendored::date::sys_days epoch = arrow_vendored::date::sys_days{arrow_vendored::date::jan / 1 / 1970};
   auto asString = arrow_vendored::date::format("%F", epoch + std::chrono::nanoseconds{date});
   return lingodb::runtime::VarLen32::fromString(asString);
}

extern "C" lingodb::runtime::VarLen32 createVarLen32(uint8_t* ptr, uint32_t len) { //NOLINT(clang-diagnostic-return-type-c-linkage)
   return lingodb::runtime::VarLen32(ptr, len);
}