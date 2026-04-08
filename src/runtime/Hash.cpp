#include "lingodb/runtime/Hash.h"

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/storage/TableStorage.h"

#include "llvm/Support/xxhash.h"

#include <arrow/array/array_binary.h>
#include <arrow/array/array_decimal.h>
#include <arrow/table.h>
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
void dbHashCombineIntArrowBatch(uint64_t* running, const arrow::Array& array, int64_t num_rows, bool isFirstColumn) {
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

void hashColumnPieceBatchRuntime(const catalog::Column& column, const arrow::Array& array, int64_t num_rows, uint64_t* running, bool isFirstColumn) {
   using catalog::LogicalTypeId;
   auto logicalType = column.getLogicalType();
   switch (logicalType.getTypeId()) {
      case LogicalTypeId::BOOLEAN: {
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
      case LogicalTypeId::INT: {
         auto intInfo = logicalType.getInfo<catalog::IntTypeInfo>();
         const size_t w = intInfo->getBitWidth();
         switch (w) {
            case 8:
               dbHashCombineIntArrowBatch<arrow::Int8Array>(running, array, num_rows, isFirstColumn);
               return;
            case 16:
               dbHashCombineIntArrowBatch<arrow::Int16Array>(running, array, num_rows, isFirstColumn);
               return;
            case 32:
               dbHashCombineIntArrowBatch<arrow::Int32Array>(running, array, num_rows, isFirstColumn);
               return;
            case 64:
               dbHashCombineIntArrowBatch<arrow::Int64Array>(running, array, num_rows, isFirstColumn);
               return;
            default:
               throw std::runtime_error("hashColumnPieceBatchRuntime: unsupported integer width");
         }
      }
      case LogicalTypeId::FLOAT: {
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
      case LogicalTypeId::DOUBLE: {
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
      case LogicalTypeId::DECIMAL: {
         const auto& a = static_cast<const arrow::Decimal128Array&>(array);
         const uint32_t precision = static_cast<uint32_t>(logicalType.getInfo<catalog::DecimalTypeInfo>()->getPrecision());
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
      case LogicalTypeId::DATE: {
         auto unit = logicalType.getInfo<catalog::DateTypeInfo>()->getUnit();
         if (unit == catalog::DateTypeInfo::DateUnit::DAY) {
            const auto& a = static_cast<const arrow::Date32Array&>(array);
            constexpr int64_t kNanosPerDay = 86400000000000ll;
            for (int64_t i = 0; i < num_rows; ++i) {
               if (array.IsNull(i)) {
                  continue;
               }
               const int64_t nanos = static_cast<int64_t>(a.Value(i)) * kNanosPerDay;
               dbHashFoldPiece(running[i], dbHash64(nanos), isFirstColumn);
            }
         } else {
            const auto& a = static_cast<const arrow::Date64Array&>(array);
            constexpr int64_t kNanosPerMilli = 1000000ll;
            for (int64_t i = 0; i < num_rows; ++i) {
               if (array.IsNull(i)) {
                  continue;
               }
               const int64_t nanos = static_cast<int64_t>(a.Value(i)) * kNanosPerMilli;
               dbHashFoldPiece(running[i], dbHash64(nanos), isFirstColumn);
            }
         }
         return;
      }
      case LogicalTypeId::TIMESTAMP: {
         const auto& a = static_cast<const arrow::TimestampArray&>(array);
         int64_t multiplier = 1;
         auto unit = logicalType.getInfo<catalog::TimestampTypeInfo>()->getUnit();
         switch (unit) {
            case catalog::TimestampTypeInfo::TimestampUnit::SECONDS:
               multiplier = 1000000000ll;
               break;
            case catalog::TimestampTypeInfo::TimestampUnit::MILLIS:
               multiplier = 1000000ll;
               break;
            case catalog::TimestampTypeInfo::TimestampUnit::MICROS:
               multiplier = 1000ll;
               break;
            case catalog::TimestampTypeInfo::TimestampUnit::NANOS:
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
      case LogicalTypeId::INTERVAL: {
         auto unit = logicalType.getInfo<catalog::IntervalTypeInfo>()->getUnit();
         if (unit == catalog::IntervalTypeInfo::IntervalUnit::MONTH) {
            dbHashCombineIntArrowBatch<arrow::MonthIntervalArray>(running, array, num_rows, isFirstColumn);
            return;
         } else {
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
         }
         return;
      }
      case LogicalTypeId::CHAR: {
         if (logicalType.getInfo<catalog::CharTypeInfo>()->getLength() == 1) {
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
         [[fallthrough]];
      }
      case LogicalTypeId::STRING: {
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
         throw std::runtime_error("hashColumnPieceBatchRuntime: unsupported logical type for hash index");
   }
}

} // namespace



void dbHashApplyColumn(std::vector<uint64_t>& running, const catalog::Column& col, const arrow::Array& arr, int64_t num_rows, bool isFirstColumn) {
   assert(static_cast<int64_t>(running.size()) == num_rows);
   hashColumnPieceBatchRuntime(col, arr, num_rows, running.data(), isFirstColumn);
}

} // namespace lingodb::runtime
