#ifndef LINGODB_RUNTIME_ARROWVIEW_H
#define LINGODB_RUNTIME_ARROWVIEW_H
#include <array>
#include <cstddef>
#include <cstdint>

namespace lingodb::runtime {
struct ArrayView {
   static constexpr int64_t maxNullCount = 1 << 20;
   static constexpr int64_t validDataLength = maxNullCount / 8;
   static std::array<uint8_t, validDataLength> validData;
   // Array data description
   int64_t length;
   int64_t nullCount;
   int64_t offset;
   int64_t nBuffers;
   int64_t nChildren;
   const void** buffers;
   const ArrayView** children;
};

struct BatchView {
   static std::array<uint16_t, 65536> defaultSelectionVector;
   static constexpr size_t maxBatchSize = 65536;
   int64_t length;
   int64_t offset;
   uint16_t* selectionVector;
   const ArrayView** arrays;
};
} // namespace lingodb::runtime
#endif //LINGODB_RUNTIME_ARROWVIEW_H
