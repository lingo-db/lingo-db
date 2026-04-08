#include "lingodb/runtime/Hash.h"

#include "llvm/Support/xxhash.h"

#include <arrow/array/array_binary.h>
#include <arrow/array/array_decimal.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/decimal.h>

#include <cassert>

EXPORT uint64_t hashVarLenData(lingodb::runtime::VarLen32 str) {
   llvm::ArrayRef<uint8_t> data(str.getPtr(), str.getLen());
   return llvm::xxHash64(data);
}

namespace lingodb::runtime {
namespace {

uint64_t byteSwap64(uint64_t x) {
   return __builtin_bswap64(x);
}

uint64_t dbHash64(int64_t v) {
   uint64_t m1 = 11400714819323198549ull * static_cast<uint64_t>(v);
   return m1 ^ byteSwap64(m1);
}

uint64_t dbHashCombineUtil(uint64_t h1, uint64_t h2) {
   return h2 ^ byteSwap64(h1);
}

void dbHashFoldPiece(uint64_t& acc, uint64_t piece, bool isFirstColumn) {
   if (isFirstColumn) {
      acc = piece;
      return;
   }
   acc ^= byteSwap64(piece);
}

template <typename TArrowArray>
void dbHashCombineIntArrowBatch(std::vector<uint64_t>& running, const arrow::Array& array, int64_t num_rows, bool isFirstColumn) {
   const auto& a = static_cast<const TArrowArray&>(array);
   for (int64_t i = 0; i < num_rows; ++i) {
      if (array.IsNull(i)) {
         continue;
      }
      dbHashFoldPiece(running[i], dbHash64(a.Value(i)), isFirstColumn);
   }
}

uint64_t dbHashVarLen32(VarLen32 v) {
   __int128 raw = v.asI128();
   uint64_t first64 = static_cast<uint64_t>(raw);
   uint64_t last64 = static_cast<uint64_t>(static_cast<unsigned __int128>(raw) >> 64);
   uint32_t len = static_cast<uint32_t>(first64 & 0xFFFFFFFFu);
   if (len > 12) {
      return hashVarLenData(v);
   }
   uint64_t fHash = dbHash64(static_cast<int64_t>(first64));
   uint64_t lHash = dbHash64(static_cast<int64_t>(last64));
   return dbHashCombineUtil(fHash, lHash);
}

void hashColumnPieceBatchRuntime(const arrow::Array& array, int64_t num_rows, std::vector<uint64_t>& running, bool isFirstColumn) {
   switch (array.type_id()) {
      case arrow::Type::type::BOOL: {
         const auto& a = static_cast<const arrow::BooleanArray&>(array);
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            int64_t b = a.Value(i) ? -1 : 0;
            dbHashFoldPiece(running[i], dbHash64(b), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::INT8: {
         dbHashCombineIntArrowBatch<arrow::Int8Array>(running, array, num_rows, isFirstColumn);
         return;
      }
      case arrow::Type::type::INT16: {
         dbHashCombineIntArrowBatch<arrow::Int16Array>(running, array, num_rows, isFirstColumn);
         return;
      }
      case arrow::Type::type::INT32: {
         dbHashCombineIntArrowBatch<arrow::Int32Array>(running, array, num_rows, isFirstColumn);
         return;
      }
      case arrow::Type::type::INT64: {
         dbHashCombineIntArrowBatch<arrow::Int64Array>(running, array, num_rows, isFirstColumn);
         return;
      }
      case arrow::Type::type::FLOAT: {
         const auto& a = static_cast<const arrow::FloatArray&>(array);
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            float f = a.Value(i);
            int32_t& fAsInt = reinterpret_cast<int32_t&>(f);
            dbHashFoldPiece(running[i], dbHash64(fAsInt), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::DOUBLE: {
         const auto& a = static_cast<const arrow::DoubleArray&>(array);
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            double d = a.Value(i);
            int32_t& dAsInt = reinterpret_cast<int32_t&>(d);
            dbHashFoldPiece(running[i], dbHash64(dAsInt), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::DECIMAL128: {
         const auto& type = static_cast<const arrow::Decimal128Type&>(*array.type());
         auto precision = type.precision();
         const auto& a = static_cast<const arrow::Decimal128Array&>(array);
         // Match compilation/lowering:
         // - precision <= 18: decimal is represented as int64
         // - precision > 18: decimal is represented as int128 and hashed as two 64-bit parts (high then low)
         if (precision <= 18) {
            for (int64_t i = 0; i < num_rows; ++i) {
               if (array.IsNull(i)) {
                  continue;
               }
               arrow::Decimal128 d(a.GetValue(i));
               dbHashFoldPiece(running[i], dbHash64(d.low_bits()), isFirstColumn);
            }
         } else {
            for (int64_t i = 0; i < num_rows; ++i) {
               if (array.IsNull(i)) {
                  continue;
               }
               arrow::Decimal128 d(a.GetValue(i));
               int64_t lowBits = d.low_bits();
               int64_t highBits = d.high_bits();
               dbHashFoldPiece(running[i], dbHash64(highBits), isFirstColumn);
               dbHashFoldPiece(running[i], dbHash64(lowBits), false);
            }
         }
         return;
      }
      case arrow::Type::type::DATE32: {
         const auto& a = static_cast<const arrow::Date32Array&>(array);
         constexpr int64_t kNanosPerDay = 86400000000000ll;
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            const int64_t nanos = static_cast<int64_t>(a.Value(i)) * kNanosPerDay;
            dbHashFoldPiece(running[i], dbHash64(nanos), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::DATE64: {
         const auto& a = static_cast<const arrow::Date64Array&>(array);
         constexpr int64_t kNanosPerMilli = 1000000ll;
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            const int64_t nanos = static_cast<int64_t>(a.Value(i)) * kNanosPerMilli;
            dbHashFoldPiece(running[i], dbHash64(nanos), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::TIMESTAMP: {
         const auto& type = static_cast<const arrow::TimestampType&>(*array.type());
         auto unit = type.unit();
         const auto& a = static_cast<const arrow::TimestampArray&>(array);
         int64_t multiplier = 1;
         switch (unit) {
            case arrow::TimeUnit::SECOND:
               multiplier = 1000000000ll;
               break;
            case arrow::TimeUnit::MILLI:
               multiplier = 1000000ll;
               break;
            case arrow::TimeUnit::MICRO:
               multiplier = 1000ll;
               break;
            case arrow::TimeUnit::NANO:
               multiplier = 1;
               break;
         }
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            const int64_t nanos = static_cast<int64_t>(a.Value(i)) * multiplier;
            dbHashFoldPiece(running[i], dbHash64(nanos), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::INTERVAL_MONTHS: {
         dbHashCombineIntArrowBatch<arrow::MonthIntervalArray>(running, array, num_rows, isFirstColumn);
         return;
      }
      case arrow::Type::type::INTERVAL_DAY_TIME: {
         const auto& a = static_cast<const arrow::DayTimeIntervalArray&>(array);
         constexpr int64_t kNanosPerDay = 86400000000000ll;
         constexpr int64_t kNanosPerMilli = 1000000ll;
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            auto v = a.Value(i);
            const int64_t nanos = static_cast<int64_t>(v.days) * kNanosPerDay + static_cast<int64_t>(v.milliseconds) * kNanosPerMilli;
            dbHashFoldPiece(running[i], dbHash64(nanos), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::FIXED_SIZE_BINARY: {
         const auto& a = static_cast<const arrow::FixedSizeBinaryArray&>(array);
         assert(a.byte_width() == 4);
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               continue;
            }
            const uint8_t* ptr = a.GetValue(i);
            const uint32_t u = (static_cast<uint32_t>(ptr[3]) << 24) | (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[1]) << 8) | static_cast<uint32_t>(ptr[0]);
            int32_t ext = static_cast<int32_t>(u);
            dbHashFoldPiece(running[i], dbHash64(ext), isFirstColumn);
         }
         return;
      }
      case arrow::Type::type::STRING: {
         const auto& a = static_cast<const arrow::StringArray&>(array);
         for (int64_t i = 0; i < num_rows; ++i) {
            if (array.IsNull(i)) {
               dbHashFoldPiece(running[i], dbHashVarLen32(VarLen32()), isFirstColumn);
            } else {
               std::string_view sv = a.GetView(i);
               VarLen32 vl(reinterpret_cast<const uint8_t*>(sv.data()), static_cast<uint32_t>(sv.size()));
               dbHashFoldPiece(running[i], dbHashVarLen32(vl), isFirstColumn);
            }
         }
         return;
      }
      default:
         throw std::runtime_error("hashColumnPieceBatchRuntime: unsupported arrow type for hash index");
   }
}

} // namespace



void dbHashApplyColumn(std::vector<uint64_t>& running, const arrow::Array& arr, bool isFirstColumn) {
   assert(static_cast<int64_t>(running.size()) == arr.length());
   hashColumnPieceBatchRuntime(arr, arr.length(), running, isFirstColumn);
}

} // namespace lingodb::runtime
