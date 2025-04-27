#ifndef LINGODB_RUNTIME_ARROW_H
#define LINGODB_RUNTIME_ARROW_H
#include <array>
#include <cstdint>

namespace lingodb::runtime {
struct ArrayView {
   static std::array<uint8_t, 4096> validData;
   // Array data description
   int64_t length;
   int64_t null_count;
   int64_t offset;
   int64_t n_buffers;
   int64_t n_children;
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
}
#endif //LINGODB_RUNTIME_ARROW_H
