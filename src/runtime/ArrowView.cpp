#include "lingodb/runtime/ArrowView.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
namespace {
static constexpr std::array<uint8_t, lingodb::runtime::ArrayView::validDataLength> createValidData() {
   std::array<uint8_t, lingodb::runtime::ArrayView::validDataLength> res;
   for (size_t i = 0; i < lingodb::runtime::ArrayView::validDataLength; i++) {
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
std::array<uint8_t, lingodb::runtime::ArrayView::validDataLength> lingodb::runtime::ArrayView::validData = createValidData();
std::array<uint16_t, 65536> lingodb::runtime::BatchView::defaultSelectionVector = createDefaultSelectionVector();

void lingodb::runtime::BatchView::printColumn(lingodb::runtime::BatchView& batch, size_t columnId) {
   if (!batch.arrays) {
      std::cerr << "BatchView::printColumn: batch has no arrays\n";
      return;
   }
   const ArrayView* array = batch.arrays[columnId];
   if (!array) {
      std::cerr << "BatchView::printColumn: column " << columnId << " is null\n";
      return;
   }

   auto isValid = [&](size_t rowInBatch) {
      if (array->nullCount == 0 || !array->buffers || !array->buffers[0]) {
         return true;
      }
      auto* validData = reinterpret_cast<const uint8_t*>(array->buffers[0]);
      const size_t absoluteIndex = batch.offset + rowInBatch + array->offset;
      return ((validData[absoluteIndex / 8] >> (absoluteIndex % 8)) & 1) != 0;
   };

   auto rowIndex = [&](size_t logicalRow) {
      if (batch.selectionVector) {
         return static_cast<size_t>(batch.selectionVector[logicalRow]);
      }
      return logicalRow;
   };

   std::cerr << "Column " << columnId << " (length=" << batch.length << ")\n";
   const size_t rowsToPrint = static_cast<size_t>(batch.length) < 5 ? static_cast<size_t>(batch.length) : 5;

   if (!array->buffers || array->nBuffers < 2 || !array->buffers[1]) {
      for (size_t i = 0; i < rowsToPrint; i++) {
         auto idx = rowIndex(i);
         std::cerr << i << ": <unsupported-layout row=" << idx << ">\n";
      }
      return;
   }

   auto* data = reinterpret_cast<const int32_t*>(array->buffers[1]);
   for (size_t i = 0; i < rowsToPrint; i++) {
      const size_t idx = rowIndex(i);
      std::cout << i << ": ";
      if (!isValid(idx)) {
         std::cout << "NULL\n";
         continue;
      }
      const size_t absoluteIndex = batch.offset + idx + array->offset;
      std::cout << data[absoluteIndex] << "\n";
   }
}
