#include "lingodb/vector/Vector.h"
#include <cstring>

namespace lingodb::vector {

void ColumnVector::densify(const std::vector<uint16_t>& selected) {
   //move the null indicators
   if (null.has_value()) {
      auto *nullArray=null.value().getPtr();
      for (size_t i = 0; i < selected.size(); i++) {
         nullArray[i]=nullArray[selected[i]];
      }
   }

   //move the raw data
   for (size_t i = 0; i < selected.size(); i++) {
      //todo: specialize for sizes (e.g., 1,2,4,8,16 bytes?)
      std::byte* to = &data.getPtr()[i * elementSize];
      std::byte* from = &data.getPtr()[selected[i] * elementSize];
      std::memcpy(to, from, elementSize);
   }
}
void Vector::densify() {
   for (auto& column : columns) {
      column.densify(selected);
   }
   dense = true;
}
} //namespace lingodb::vector