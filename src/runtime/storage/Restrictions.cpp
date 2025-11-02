#include "lingodb/runtime/storage/Restrictions.h"
#include "arrow/table.h"
#include "lingodb/runtime/ArrowView.h"
#include <cstddef>
#include <cstring>
#include <regex>
#include <arrow/util/decimal.h>
#include <arrow/util/value_parsing.h>
namespace {
int32_t parseDate32(std::string str) {
   static std::regex r("(\\d\\d\\d\\d)-(\\d)-(\\d\\d)");
   str = std::regex_replace(str, r, "$1-0$2-$3");
   int32_t res;
   if (!arrow::internal::ParseValue<arrow::Date32Type>(str.data(), str.length(), &res)) {
      throw std::runtime_error("could not parse date");
   }
   return res;
}
}
template <class T>
struct Eq {
   static bool apply(T a, T b) {
      return a == b;
   }
};
template <class T>
struct Neq {
   static bool apply(T a, T b) {
      return a != b;
   }
};
template <class T>
struct Gt {
   static bool apply(T a, T b) {
      return a > b;
   }
};
template <class T>
struct GtE {
   static bool apply(T a, T b) {
      return a >= b;
   }
};
template <class T>
struct Lt {
   static bool apply(T a, T b) {
      return a < b;
   }
};
template <class T>
struct LtE {
   static bool apply(T a, T b) {
      return a <= b;
   }
};

template <class T, template <class> class CMP>
class SimpleTypeFilter : public lingodb::runtime::Filter {
   T value;

   public:
   SimpleTypeFilter(T value) : value(value) {}
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      const T* data = reinterpret_cast<const T*>(arrayView->buffers[1]) + offset + arrayView->offset;
      size_t len4 = len & ~3;
      auto* writer = nextSelVec;
      for (size_t i = 0; i < len4; i += 4) {
         size_t index0 = currSelVec[i];
         size_t index1 = currSelVec[i + 1];
         size_t index2 = currSelVec[i + 2];
         size_t index3 = currSelVec[i + 3];
         *writer = index0;
         writer += CMP<T>::apply(data[index0], value);
         *writer = index1;
         writer += CMP<T>::apply(data[index1], value);
         *writer = index2;
         writer += CMP<T>::apply(data[index2], value);
         *writer = index3;
         writer += CMP<T>::apply(data[index3], value);
      }
      for (size_t i = len4; i < len; i++) {
         size_t index0 = currSelVec[i];
         *writer = index0;
         writer += CMP<T>::apply(data[index0], value);
      }
      return writer - nextSelVec;
   }
};
template <template <class> class CMP>
class VarLen32Filter : public lingodb::runtime::Filter {
   std::string value;

   public:
   VarLen32Filter(std::string value) : value(value) {}
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      const uint8_t* data = reinterpret_cast<const uint8_t*>(arrayView->buffers[2]);
      const int32_t* offsets = reinterpret_cast<const int32_t*>(arrayView->buffers[1]) + offset + arrayView->offset;
      auto* writer = nextSelVec;
      size_t len4 = len & ~3;
      for (size_t i = 0; i < len4; i+=4) {
         size_t index0 = currSelVec[i];
         size_t index1 = currSelVec[i + 1];
         size_t index2 = currSelVec[i + 2];
         size_t index3 = currSelVec[i + 3];
         int32_t offset0 = offsets[index0];
         int32_t nextOffset0 = offsets[index0 + 1];
         std::string_view strView0(reinterpret_cast<const char*>(data + offset0), nextOffset0 - offset0);
         *writer = index0;
         writer += CMP<std::string_view>::apply(strView0, value);
         int32_t offset1 = offsets[index1];
         int32_t nextOffset1 = offsets[index1 + 1];
         std::string_view strView1(reinterpret_cast<const char*>(data + offset1), nextOffset1 - offset1);
         *writer = index1;
         writer += CMP<std::string_view>::apply(strView1, value);
         int32_t offset2 = offsets[index2];
         int32_t nextOffset2 = offsets[index2 + 1];
         std::string_view strView2(reinterpret_cast<const char*>(data + offset2), nextOffset2 - offset2);
         *writer = index2;
         writer += CMP<std::string_view>::apply(strView2, value);
         int32_t offset3 = offsets[index3];
         int32_t nextOffset3 = offsets[index3 + 1];
         std::string_view strView3(reinterpret_cast<const char*>(data + offset3), nextOffset3 - offset3);
         *writer = index3;
         writer += CMP<std::string_view>::apply(strView3, value);
      }
      for (size_t i = len4; i < len; i++) {
         size_t currOffset = offsets[currSelVec[i]];
         size_t nextOffset = offsets[currSelVec[i] + 1];
         std::string_view strView(reinterpret_cast<const char*>(data + currOffset), nextOffset - currOffset);
         *writer = currSelVec[i];
         writer += CMP<std::string_view>::apply(strView, value);
      }
      return writer - nextSelVec;
   }
};
template <class T>
std::unique_ptr<lingodb::runtime::Filter> createSimpleTypeFilter(lingodb::runtime::FilterOp op, T value) {
   switch (op) {
      case lingodb::runtime::FilterOp::LT:
         return std::make_unique<SimpleTypeFilter<T, Lt>>(value);
      case lingodb::runtime::FilterOp::LTE:
         return std::make_unique<SimpleTypeFilter<T, LtE>>(value);
      case lingodb::runtime::FilterOp::GT:
         return std::make_unique<SimpleTypeFilter<T, Gt>>(value);
      case lingodb::runtime::FilterOp::GTE:
         return std::make_unique<SimpleTypeFilter<T, GtE>>(value);
      case lingodb::runtime::FilterOp::EQ:
         return std::make_unique<SimpleTypeFilter<T, Eq>>(value);
      case lingodb::runtime::FilterOp::NEQ:
         return std::make_unique<SimpleTypeFilter<T, Neq>>(value);
      default:
         throw std::runtime_error("unsupported filter op");
   }
}
std::pair<size_t, uint16_t*> lingodb::runtime::Restrictions::applyFilters(size_t offset, size_t length, uint16_t* selVec1, uint16_t* selVec2, std::function<const ArrayView*(size_t)> getArrayView) {
   uint16_t* currentSelVec = selVec1;
   std::memcpy(currentSelVec, lingodb::runtime::BatchView::defaultSelectionVector.data(), length * sizeof(uint16_t));
   uint16_t* nextSelVec = selVec2;
   size_t currentLen = length;
   for (auto& filterPair : filters) {
      auto& filter = filterPair.first;
      size_t colId = filterPair.second;
      const lingodb::runtime::ArrayView* arrayView = getArrayView(colId);
      currentLen = filter->filter(currentLen, currentSelVec, nextSelVec, arrayView, offset);
      std::swap(currentSelVec, nextSelVec);
   }
   assert(currentSelVec[currentLen - 1] < length);
   return {currentLen, currentSelVec};
}

std::unique_ptr<lingodb::runtime::Restrictions> lingodb::runtime::Restrictions::create(std::vector<lingodb::runtime::FilterDescription> filterDescs, const arrow::Schema& schema) {
   auto restrictions = std::make_unique<lingodb::runtime::Restrictions>();
   for (auto& filterDesc : filterDescs) {
      size_t colId = schema.GetFieldIndex(filterDesc.columnName);
      if (colId == static_cast<size_t>(-1)) {
         throw std::runtime_error("unknown column in filter");
      }
      auto type = schema.field(colId)->type();
      switch (type->id()) {
         case arrow::Type::FIXED_SIZE_BINARY: {
            auto fixedSizedType = std::static_pointer_cast<arrow::FixedSizeBinaryType>(type);
            if (fixedSizedType->byte_width() == 4) {
               std::string strVal = std::get<std::string>(filterDesc.value);
               int32_t intVal = 0;
               std::memcpy(&intVal, strVal.data(), std::min(sizeof(intVal), strVal.size()));
               restrictions->filters.push_back({createSimpleTypeFilter<int32_t>(filterDesc.op, intVal), colId});
               break;
            } else {
               throw std::runtime_error("unsupported fixed size binary width in filter");
            }
            break;
         }
         case arrow::Type::INT32: {
            auto intVal = static_cast<int32_t>(std::get<int64_t>(filterDesc.value));
            restrictions->filters.push_back({createSimpleTypeFilter<int32_t>(filterDesc.op, intVal), colId});
            break;
         }
         case arrow::Type::DATE32: {
            auto stringVal = std::get<std::string>(filterDesc.value);
            auto intVal = parseDate32(stringVal);
            restrictions->filters.push_back({createSimpleTypeFilter<int32_t>(filterDesc.op, intVal), colId});
            break;
         }
         case arrow::Type::DECIMAL128: {
            auto decimalType = std::static_pointer_cast<arrow::Decimal128Type>(type);
            __int128 decimalValue;
            if (std::holds_alternative<std::string>(filterDesc.value)) {
               int32_t precision;
               int32_t scale;
               arrow::Decimal128 decimalrep;
               if (!arrow::Decimal128::FromString(std::get<std::string>(filterDesc.value), &decimalrep, &precision, &scale).ok()) {
                  assert(false && "could not parse decimal const");
               }
               auto x = decimalrep.Rescale(scale, decimalType->scale());
               decimalrep = x.ValueOrDie();
               decimalValue = *reinterpret_cast<const __int128*>(decimalrep.native_endian_bytes());
            } else if (std::holds_alternative<int64_t>(filterDesc.value)) {
               decimalValue = static_cast<__int128>(std::get<int64_t>(filterDesc.value));
               int32_t scale = decimalType->scale();
               while (scale > 0) {
                  decimalValue *= 10;
                  scale--;
               }
            } else {
               throw std::runtime_error("unsupported decimal constant type");
            }
            restrictions->filters.push_back({createSimpleTypeFilter<__int128>(filterDesc.op, decimalValue), colId});
            break;
         }
         case arrow::Type::STRING: {
            std::string value = std::get<std::string>(filterDesc.value);
            switch (filterDesc.op) {
               case lingodb::runtime::FilterOp::LT:
                  restrictions->filters.push_back({std::make_unique<VarLen32Filter<Lt>>(value), colId});
                  break;
               case lingodb::runtime::FilterOp::LTE:
                  restrictions->filters.push_back({std::make_unique<VarLen32Filter<LtE>>(value), colId});
                  break;
               case lingodb::runtime::FilterOp::GT:
                  restrictions->filters.push_back({std::make_unique<VarLen32Filter<Gt>>(value), colId});
                  break;
               case lingodb::runtime::FilterOp::GTE:
                  restrictions->filters.push_back({std::make_unique<VarLen32Filter<GtE>>(value), colId});
                  break;
               case lingodb::runtime::FilterOp::EQ:
                  restrictions->filters.push_back({std::make_unique<VarLen32Filter<Eq>>(value), colId});
                  break;
               case lingodb::runtime::FilterOp::NEQ:
                  restrictions->filters.push_back({std::make_unique<VarLen32Filter<Neq>>(value), colId});
                  break;
               default:
                  throw std::runtime_error("unsupported filter op for string");
            }
            break;
         }
         default:
            throw std::runtime_error("unsupported type in filter" + type->ToString());
      }
   }
   return restrictions;
}