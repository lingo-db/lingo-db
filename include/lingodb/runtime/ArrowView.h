#ifndef LINGODB_RUNTIME_ARROWVIEW_H
#define LINGODB_RUNTIME_ARROWVIEW_H
#include <array>
#include <cstdint>

namespace lingodb::runtime {
struct ArrayView {
   static std::array<uint8_t, 4096> validData;
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
   int64_t length;
   int64_t offset;
   int16_t* selectionVector;
   const ArrayView** arrays;
};
} // namespace lingodb::runtime
#endif //LINGODB_RUNTIME_ARROWVIEW_H
