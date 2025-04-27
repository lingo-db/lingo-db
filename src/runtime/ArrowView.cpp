#include "lingodb/runtime/ArrowView.h"
#include<cstddef>
namespace {
static constexpr std::array<uint8_t, 4096> createValidData() {
   std::array<uint8_t, 4096> res;
   for (size_t i = 0; i < 4096; i++) {
      res[i] = 0xff;
   }
   return res;
}
static constexpr std::array<uint16_t, 65536> createDefaultSelectionVector() {
   std::array<uint16_t, 65536> res;
   for (size_t i = 0; i < 65536; i++) {
      res[i] = i;
   }
   return res;
}
} // namespace
std::array<uint8_t, 4096> lingodb::runtime::ArrayView::validData = createValidData();
std::array<uint16_t, 65536> lingodb::runtime::BatchView::defaultSelectionVector = createDefaultSelectionVector();
