#ifndef LINGODB_VECTOR_VECTOR_H
#define LINGODB_VECTOR_VECTOR_H
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
namespace lingodb::vector {
class RawData {
   std::byte* ptr;
   bool deallocate;

   public:
   RawData(size_t bytes) {
      ptr = new std::byte[bytes];
      deallocate = true;
   }
   RawData(std::byte* ptr) : ptr(ptr), deallocate(false) {}
   RawData(RawData&& other) {
      this->ptr = other.ptr;
      this->deallocate = other.deallocate;
      other.deallocate = false;
      other.ptr = nullptr;
   }
   std::byte* getPtr() {
      return ptr;
   }
   ~RawData() {
      if (deallocate) {
         delete[] ptr;
      }
   }
};
class ColumnVector {
   std::optional<RawData> null;
   RawData data;
   size_t elementSize;

   public:
   void densify(const std::vector<uint16_t>& selected);
};
class Vector {
   public:
   static constexpr size_t maxSize=1024;
   // is every row in the current vector valid? dense=true: yes, dense=false: no
   bool dense;
   //row indexes that are valid, only interesting if dense=false.
   std::vector<uint16_t> selected;
   //vector data for the columns
   std::vector<ColumnVector> columns;

   void densify();
};

class Expression{
   ColumnVector computed;
   //(re-)compute the `computed` data using the vector
   virtual void update(Vector& data)=0;
};

} // namespace lingodb::vector

#endif //LINGODB_VECTOR_VECTOR_H
