#include "runtime/helpers.h"

#define STR_CMP(OP, NAME)                                                                                  \
   extern "C" bool NAME##_utf8_utf8(char* str1, uint32_t len1, char* str2, uint32_t len2);                 \
   extern "C" bool _mlir_ciface_cmp_string_##OP(bool null, runtime::String* str1, runtime::String* str2) { \
      if (null) {                                                                                          \
         return false;                                                                                     \
      } else {                                                                                             \
         return NAME##_utf8_utf8((*str1).data(), (*str1).len(), (*str2).data(), (*str2).len());            \
      }                                                                                                    \
   }

STR_CMP(eq, equal)
STR_CMP(neq, not_equal)
STR_CMP(lt, less_than)
STR_CMP(lte, less_than_or_equal_to)
STR_CMP(gt, greater_than)
STR_CMP(gte, greater_than_or_equal_to)

