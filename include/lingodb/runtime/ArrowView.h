#ifndef LINGODB_RUNTIME_ARROWVIEW_H
#define LINGODB_RUNTIME_ARROWVIEW_H
#include <array>
#include <cstddef>
#include <cstdint>

namespace lingodb::runtime {
struct ArrayView {
   static constexpr int32_t maxNullCount = 1 << 20;
   static constexpr int32_t validDataLength = maxNullCount / 8;
   static std::array<uint8_t, validDataLength> validData;
   // Array data description
   int32_t length;
   int32_t nullCount;
   int32_t offset;
   int32_t nBuffers;
   int32_t nChildren;
   const void** buffers;
   const ArrayView** children;
};

struct BatchView {
   static std::array<uint16_t, 65536> defaultSelectionVector;
   static constexpr size_t maxBatchSize = 65536;
   int32_t length;
   int32_t offset;
   uint16_t* selectionVector;
   const ArrayView** arrays;
};
} // namespace lingodb::runtime
#endif //LINGODB_RUNTIME_ARROWVIEW_H
