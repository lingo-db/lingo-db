#include "lingodb/runtime/storage/Restrictions.h"

#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/LazyJoinHashtable.h"
#include "lingodb/runtime/SIP.h"

#include <llvm/Support/xxhash.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <regex>

#include <arrow/table.h>
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
// does not need to look in the currSelVec, -> directly iterate over bitmap
// first loop: make sure that we are aligned to byte boundary
// second loop: process 8 at a time
// last loop: process remaining elements
// again: currSelVector is just 0 to len-1, so we can directly iterate over the validity bitmap
class FirstNotNullFilter : public lingodb::runtime::Filter {
   public:
   FirstNotNullFilter() {}
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      if (arrayView->nullCount == 0) {
         //fast path: no nulls
         std::memcpy(nextSelVec, currSelVec, len * sizeof(uint16_t));
         return len;
      }
      const uint8_t* validData = reinterpret_cast<const uint8_t*>(arrayView->buffers[0]);
      auto* writer = nextSelVec;
      size_t i = 0;
      //align to byte boundary
      for (; i < len && ((i + offset + arrayView->offset) % 8) != 0; i++) {
         size_t index0 = i + offset + arrayView->offset;
         *writer = i;
         writer += (bool) ((validData[index0 / 8] >> (index0 % 8)) & 1);
      }
      assert((i + offset + arrayView->offset) % 8 == 0 || i == len);
      size_t len8 = len & ~7;
      for (; i < len8; i += 8) {
         uint8_t byte = validData[(i + offset + arrayView->offset) / 8];
         *writer = i;
         writer += (bool) (byte & 1);
         *writer = i + 1;
         writer += (bool) (byte & 2);
         *writer = i + 2;
         writer += (bool) (byte & 4);
         *writer = i + 3;
         writer += (bool) (byte & 8);
         *writer = i + 4;
         writer += (bool) (byte & 16);
         *writer = i + 5;
         writer += (bool) (byte & 32);
         *writer = i + 6;
         writer += (bool) (byte & 64);
         *writer = i + 7;
         writer += (bool) (byte & 128);
      }
      for (; i < len; i++) {
         size_t index0 = i + offset + arrayView->offset;
         *writer = i;
         writer += (bool) ((validData[index0 / 8] >> (index0 % 8)) & 1);
      }
      return writer - nextSelVec;
   }
};

class NotNullFilter : public lingodb::runtime::Filter {
   public:
   NotNullFilter() {}
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      if (arrayView->nullCount == 0) {
         //fast path: no nulls
         std::memcpy(nextSelVec, currSelVec, len * sizeof(uint16_t));
         return len;
      }
      const uint8_t* validData = reinterpret_cast<const uint8_t*>(arrayView->buffers[0]);
      auto* writer = nextSelVec;
      size_t len8 = len & ~7;
      for (size_t i = 0; i < len8; i += 8) {
         size_t index0 = currSelVec[i] + offset + arrayView->offset;
         size_t index1 = currSelVec[i + 1] + offset + arrayView->offset;
         size_t index2 = currSelVec[i + 2] + offset + arrayView->offset;
         size_t index3 = currSelVec[i + 3] + offset + arrayView->offset;
         size_t index4 = currSelVec[i + 4] + offset + arrayView->offset;
         size_t index5 = currSelVec[i + 5] + offset + arrayView->offset;
         size_t index6 = currSelVec[i + 6] + offset + arrayView->offset;
         size_t index7 = currSelVec[i + 7] + offset + arrayView->offset;
         *writer = currSelVec[i];
         writer += (bool) ((validData[index0 / 8] >> (index0 % 8)) & 1);
         *writer = currSelVec[i + 1];
         writer += (bool) ((validData[index1 / 8] >> (index1 % 8)) & 1);
         *writer = currSelVec[i + 2];
         writer += (bool) ((validData[index2 / 8] >> (index2 % 8)) & 1);
         *writer = currSelVec[i + 3];
         writer += (bool) ((validData[index3 / 8] >> (index3 % 8)) & 1);
         *writer = currSelVec[i + 4];
         writer += (bool) ((validData[index4 / 8] >> (index4 % 8)) & 1);
         *writer = currSelVec[i + 5];
         writer += (bool) ((validData[index5 / 8] >> (index5 % 8)) & 1);
         *writer = currSelVec[i + 6];
         writer += (bool) ((validData[index6 / 8] >> (index6 % 8)) & 1);
         *writer = currSelVec[i + 7];
         writer += (bool) ((validData[index7 / 8] >> (index7 % 8)) & 1);
      }
      for (size_t i = len8; i < len; i++) {
         size_t index0 = currSelVec[i];
         *writer = index0;
         writer += (bool) ((validData[(index0 + offset + arrayView->offset) / 8] >> ((index0 + offset + arrayView->offset) % 8)) & 1);
      }
      return writer - nextSelVec;
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
template <class T>
class SimpleTypeInFilter : public lingodb::runtime::Filter {
   std::vector<T> values;

   public:
   SimpleTypeInFilter(std::vector<T> values) : values(values) {}
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      const T* data = reinterpret_cast<const T*>(arrayView->buffers[1]) + offset + arrayView->offset;
      auto* writer = nextSelVec;
      for (size_t i = 0; i < len; i++) {
         size_t index0 = currSelVec[i];
         for (auto value : values) {
            if (value == data[index0]) {
               *writer = index0;
               writer++;
               break;
            }
         }
      }
      return writer - nextSelVec;
   }
};
class VarLen32FilterIn : public lingodb::runtime::Filter {
   std::vector<std::string> values;

   public:
   VarLen32FilterIn(std::vector<std::string> values) : values(values) {}
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      const uint8_t* data = reinterpret_cast<const uint8_t*>(arrayView->buffers[2]);
      const int32_t* offsets = reinterpret_cast<const int32_t*>(arrayView->buffers[1]) + offset + arrayView->offset;
      auto* writer = nextSelVec;
      for (size_t i = 0; i < len; i++) {
         size_t index0 = currSelVec[i];
         int32_t offset0 = offsets[index0];
         int32_t nextOffset0 = offsets[index0 + 1];
         std::string_view strView(reinterpret_cast<const char*>(data + offset0), nextOffset0 - offset0);
         for (const auto& s : values) {
            if (strView == s) {
               *writer = index0;
               writer++;
               break;
            }
         }
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
      for (size_t i = 0; i < len4; i += 4) {
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
class BitMapFilter : public lingodb::runtime::Filter {
   size_t* bitMap;

   public:
   BitMapFilter(size_t* bitMap) {}
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      const T* data = reinterpret_cast<const T*>(arrayView->buffers[1]) + offset + arrayView->offset;
      auto* writer = nextSelVec;
      for (size_t i = 0; i < len; i++) {
         size_t index0 = currSelVec[i];
         auto index = std::hash<T>{}(data[index0]) % 64;
         //Check if bit at index in bitMap is set 0000000000000010
         auto toCheck = *bitMap;
         toCheck >> index;
         toCheck << (64 - index);
         if (toCheck >= 0) {
            *writer = index0;
            writer++;
         }
      }
      return writer - nextSelVec;
   }
};

template <class T>
class HashViewFilter : public lingodb::runtime::Filter {
   lingodb::runtime::SIP::SIPNode* sip;
   lingodb::runtime::HashIndexedView* view;
   template <class V>
   inline uint64_t hashValue(V val) {
      if constexpr (std::is_integral_v<V>) {
         size_t p1 = 11400714819323198549ull;
         size_t m1 = static_cast<size_t>(val) * p1;
         return m1 ^ __builtin_bswap64(m1);

      } else if constexpr (std::is_same_v<V, std::string_view>) {
         const uint32_t len = static_cast<uint32_t>(val.size());

         // Match util::VarLenTryCheapHash for short strings and runtime hashVarLenData for long ones.
         if (len <= lingodb::runtime::VarLen32::shortLen) {
            uint8_t buf[16] = {0};
            std::memcpy(buf, &len, sizeof(len));
            std::memcpy(buf + sizeof(len), val.data(), val.size());

            uint64_t first64;
            uint64_t last64;
            std::memcpy(&first64, buf, sizeof(first64));
            std::memcpy(&last64, buf + 8, sizeof(last64));

            auto hash64 = [](uint64_t v) {
               const uint64_t p1 = 11400714819323198549ull;
               uint64_t m1 = v * p1;
               return m1 ^ __builtin_bswap64(m1);
            };
            auto hashCombine = [](uint64_t h1, uint64_t h2) { return __builtin_bswap64(h1) ^ h2; };

            uint64_t h1 = hash64(first64);
            uint64_t h2 = hash64(last64);
            return hashCombine(h1, h2);
         }

         llvm::ArrayRef<uint8_t> data(reinterpret_cast<const uint8_t*>(val.data()), val.size());
         return llvm::xxHash64(data);
      }
      return 0;
   }

   public:
   HashViewFilter(lingodb::runtime::SIP::SIPNode* sip) : sip(sip), view(sip->hashView) {
      assert(view);
   }
   size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) override {
      if (sip->skipState.load(std::memory_order_relaxed)) {
         std::memcpy(nextSelVec, currSelVec, len * sizeof(uint16_t));
         return len;
      }
      auto* writer = nextSelVec;
      size_t filteredCount = 0;
      if constexpr (std::is_same_v<T, std::string>) {
         const uint8_t* data = reinterpret_cast<const uint8_t*>(arrayView->buffers[2]);
         const int32_t* offsets = reinterpret_cast<const int32_t*>(arrayView->buffers[1]) + offset + arrayView->offset;
         for (size_t i = 0; i < len; i++) {
            size_t index0 = currSelVec[i];
            int32_t offset0 = offsets[index0];
            int32_t nextOffset0 = offsets[index0 + 1];
            std::string_view strView0(reinterpret_cast<const char*>(data + offset0), nextOffset0 - offset0);
            auto hashed = hashValue<std::string_view>(strView0);
            lingodb::runtime::HashIndexedView::Entry* current(view->ht[hashed & view->getHtMask()]);
            bool matches = lingodb::runtime::matchesTag(current, hashed);
            //Keep value (IN filter logic - keep if found in hash table)
            *writer = index0;
            writer += matches;
            filteredCount+=!matches;
         }

      } else {
         size_t len4 = len & ~3;
         const T* data = reinterpret_cast<const T*>(arrayView->buffers[1]) + offset + arrayView->offset;
         for (size_t i = 0; i < len4; i += 4) {
            size_t index0 = currSelVec[i];
            size_t index1 = currSelVec[i + 1];
            size_t index2 = currSelVec[i + 2];
            size_t index3 = currSelVec[i + 3];
            auto hashed0 = hashValue<T>(data[index0]);
            auto hashed1 = hashValue<T>(data[index1]);
            auto hashed2 = hashValue<T>(data[index2]);
            auto hashed3 = hashValue<T>(data[index3]);
            lingodb::runtime::HashIndexedView::Entry* entry0(view->ht[hashed0 & view->getHtMask()]);
            *writer = index0;
            bool m0 = lingodb::runtime::matchesTag(entry0, hashed0);
            writer += m0;
            lingodb::runtime::HashIndexedView::Entry* entry1(view->ht[hashed1 & view->getHtMask()]);
            *writer = index1;
            bool m1 = lingodb::runtime::matchesTag(entry1, hashed1);
            writer += m1;
            lingodb::runtime::HashIndexedView::Entry* entry2(view->ht[hashed2 & view->getHtMask()]);
            *writer = index2;
            bool m2 = lingodb::runtime::matchesTag(entry2, hashed2);
            writer += m2;
            lingodb::runtime::HashIndexedView::Entry* entry3(view->ht[hashed3 & view->getHtMask()]);
            *writer = index3;
            bool m3 = lingodb::runtime::matchesTag(entry3, hashed3);
            writer += m3;
            filteredCount+=!m3 + !m2 + !m1 + !m0;

         }

         for (size_t i = len4; i < len; i++) {
            size_t index0 = currSelVec[i];
            auto hashed = hashValue<T>(data[index0]);
            lingodb::runtime::HashIndexedView::Entry* current(view->ht[hashed & view->getHtMask()]);
            bool matches = lingodb::runtime::matchesTag(current, hashed);
            *writer = index0;
            writer += matches;
            filteredCount+=!matches;

         }
      }
      if (filteredCount == 0) {
         //SKIP
         sip->skipState.fetch_add(1, std::memory_order_relaxed);
      }
#if DEBUG
      sip->filteredCount.fetch_add(filteredCount);
      sip->completeCount.fetch_add(len);
#endif
      return writer - nextSelVec;
   }
};

template <class T>
std::unique_ptr<lingodb::runtime::Filter> createSimpleTypeSimpleFilters(lingodb::runtime::FilterOp op, T value) {
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
template <class T, class PT>
std::unique_ptr<lingodb::runtime::Filter> createSimpleTypeFilter(lingodb::runtime::FilterOp op, lingodb::runtime::FilterDescription& filterDesc) {
   switch (op) {
      case lingodb::runtime::FilterOp::LT:
      case lingodb::runtime::FilterOp::LTE:
      case lingodb::runtime::FilterOp::GT:
      case lingodb::runtime::FilterOp::GTE:
      case lingodb::runtime::FilterOp::EQ:
      case lingodb::runtime::FilterOp::NEQ: {
         auto value = static_cast<T>(std::get<PT>(filterDesc.value));
         return createSimpleTypeSimpleFilters<T>(op, value);
      }
      case lingodb::runtime::FilterOp::IN: {
         auto valuesPt = (std::get<std::vector<PT>>(filterDesc.values));
         std::vector<T> values;
         for (auto v : valuesPt) values.push_back(v);
         return std::make_unique<SimpleTypeInFilter<T>>(values);
      }
      case lingodb::runtime::FilterOp::SIP: {
         auto sipId = static_cast<uint8_t>(std::get<std::int64_t>(filterDesc.value));
         return std::make_unique<HashViewFilter<T>>(lingodb::runtime::SIP::getFilterNode(sipId));
      }
      default:
         throw std::runtime_error("unsupported filter op");
   }
}
} // namespace

std::pair<size_t, uint16_t*> lingodb::runtime::Restrictions::applyFilters(size_t offset, size_t length, uint16_t* selVec1, uint16_t* selVec2, std::function<const ArrayView*(size_t)> getArrayView) {
   uint16_t* currentSelVec = selVec1;
   assert(length <= BatchView::maxBatchSize);
   std::memcpy(currentSelVec, lingodb::runtime::BatchView::defaultSelectionVector.data(), length * sizeof(uint16_t));
   uint16_t* nextSelVec = selVec2;
   size_t currentLen = length;
   for (auto& filterPair : filters) {
      auto& filter = filterPair.first;
      size_t colId = filterPair.second;
      const lingodb::runtime::ArrayView* arrayView = getArrayView(colId);
      currentLen = filter->filter(currentLen, currentSelVec, nextSelVec, arrayView, offset);
      std::swap(currentSelVec, nextSelVec);
      assert(currentLen <= length);
   }
   assert(currentLen == 0 || currentSelVec[currentLen - 1] < length);
   return {currentLen, currentSelVec};
}

std::unique_ptr<lingodb::runtime::Restrictions> lingodb::runtime::Restrictions::create(std::vector<lingodb::runtime::FilterDescription> filterDescs, const arrow::Schema& schema) {
   auto restrictions = std::make_unique<lingodb::runtime::Restrictions>();
   for (auto& filterDesc : filterDescs) {
      size_t colId = schema.GetFieldIndex(filterDesc.columnName);
      if (colId == static_cast<size_t>(-1)) {
         throw std::runtime_error("unknown column in filter: " + filterDesc.columnName + " table");
      }
      if (filterDesc.op == FilterOp::NOTNULL) {
         if (restrictions->filters.empty()) {
            restrictions->filters.push_back({std::make_unique<FirstNotNullFilter>(), colId});
         } else {
            restrictions->filters.push_back({std::make_unique<NotNullFilter>(), colId});
         }
         continue;
      }
      auto type = schema.field(colId)->type();
      switch (type->id()) {
         case arrow::Type::FIXED_SIZE_BINARY: {
            auto fixedSizedType = std::static_pointer_cast<arrow::FixedSizeBinaryType>(type);
            if (fixedSizedType->byte_width() == 4) {
               std::string strVal = std::get<std::string>(filterDesc.value);
               assert(strVal.size() <= 4);
               int32_t intVal = 0;
               std::memcpy(&intVal, strVal.data(), std::min(sizeof(intVal), strVal.size()));
               restrictions->filters.push_back({createSimpleTypeSimpleFilters<int32_t>(filterDesc.op, intVal), colId});
               break;
            } else {
               throw std::runtime_error("unsupported fixed size binary width in filter");
            }
            break;
         }
         case arrow::Type::INT8: {
            restrictions->filters.push_back({createSimpleTypeFilter<int8_t, int64_t>(filterDesc.op, filterDesc), colId});
            break;
         }
         case arrow::Type::INT16: {
            restrictions->filters.push_back({createSimpleTypeFilter<int16_t, int64_t>(filterDesc.op, filterDesc), colId});
            break;
         }

         case arrow::Type::INT32: {
            restrictions->filters.push_back({createSimpleTypeFilter<int32_t, int64_t>(filterDesc.op, filterDesc), colId});
            break;
         }
         case arrow::Type::INT64: {
            restrictions->filters.push_back({createSimpleTypeFilter<int64_t, int64_t>(filterDesc.op, filterDesc), colId});
            break;
         }
         case arrow::Type::DATE32: {
            if (filterDesc.op == FilterOp::SIP) {
               std::cerr << "SIP for Date32 not yet supported" << std::endl;
               // restrictions->filters.push_back({std::make_unique<HashViewFilter<int32_t>>(std::get<std::string>(filterDesc.value)), colId});
            } else if (filterDesc.op == FilterOp::IN) {
               std::vector<int32_t> values;
               for (auto strVal : std::get<std::vector<std::string>>(filterDesc.values)) {
                  values.push_back(parseDate32(strVal));
               }
               restrictions->filters.push_back({std::make_unique<SimpleTypeInFilter<int32_t>>(values), colId});
            } else {
               auto stringVal = std::get<std::string>(filterDesc.value);
               auto intVal = parseDate32(stringVal);
               restrictions->filters.push_back({createSimpleTypeSimpleFilters<int32_t>(filterDesc.op, intVal), colId});
            }
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
            restrictions->filters.push_back({createSimpleTypeSimpleFilters<__int128>(filterDesc.op, decimalValue), colId});
            break;
         }
         case arrow::Type::STRING: {
            std::string value;
            if (filterDesc.op != FilterOp::IN && filterDesc.op != FilterOp::SIP) {
               value = std::get<std::string>(filterDesc.value);
            }
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
               case lingodb::runtime::FilterOp::IN: {
                  auto values = std::get<std::vector<std::string>>(filterDesc.values);
                  restrictions->filters.push_back({std::make_unique<VarLen32FilterIn>(values), colId});
                  break;
               }
               case lingodb::runtime::FilterOp::SIP: {
                  auto sipId = static_cast<uint8_t>(std::get<std::int64_t>(filterDesc.value));
                  restrictions->filters.push_back({std::make_unique<HashViewFilter<std::string>>(SIP::getFilterNode(sipId)), colId});
                  break;
               }

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