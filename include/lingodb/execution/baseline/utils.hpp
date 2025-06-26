#ifndef UTILS_HPP
#define UTILS_HPP

#include <llvm/ADT/TypeSwitch.h>
#include <ranges>
#include <lingodb/compiler/Dialect/util/UtilTypes.h>

namespace lingodb::execution::baseline {
using namespace compiler;

class TupleHelper {
   mlir::TupleType tupleType;

   public:
   explicit TupleHelper(mlir::TupleType tupleType) : tupleType(tupleType) {}

   [[nodiscard]] unsigned getElementOffset(uint32_t index) const noexcept {
      size_t offset = 0;
      assert(index < tupleType.getTypes().size() && "Index out of bounds for tuple type");
      for (const mlir::Type elem : tupleType.getTypes() | std::views::take(index - 1)) {
         unsigned elemAlign = 0;
         size_t elemSize = 0;
         llvm::TypeSwitch<mlir::Type>(elem)
            .Case<dialect::util::RefType>([&](dialect::util::RefType t) {
               elemSize = sizeof(void*);
               elemAlign = alignof(void*);
            })
            .Case<mlir::IntegerType>([&](mlir::IntegerType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (elemSize) {
                  case 1: elemAlign = alignof(int8_t); break;
                  case 2: elemAlign = alignof(int16_t); break;
                  case 4: elemAlign = alignof(int32_t); break;
                  case 8: elemAlign = alignof(int64_t); break;
                  case 16: elemAlign = alignof(__int128); break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::FloatType>([&](mlir::FloatType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (t.getIntOrFloatBitWidth()) {
                  case 32:
                     static_assert(sizeof(float) == 4);
                     elemAlign = alignof(float);
                     break;
                  case 64:
                     static_assert(sizeof(double) == 8);
                     elemAlign = alignof(double);
                     break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::TupleType>([&](mlir::TupleType tupleType) {
               auto [size, align] = TupleHelper{tupleType}.sizeAndPadding();
               elemSize = size;
               elemAlign = align;
            })
            .Case<mlir::IndexType>([&](auto) {
               elemSize = sizeof(uint64_t);
               elemAlign = alignof(uint64_t);
            })
            .Default([](mlir::Type) {
               assert(false && "Cannot calculate size for unsupported type.");
               return 0;
            });
         // enforce alignment requirement
         offset += offset % elemAlign;
         // add offset of element
         offset += elemSize;
      }
      return offset;
   }

   [[nodiscard]] std::pair<size_t, unsigned> sizeAndPadding() const noexcept {
      size_t offset = 0;
      unsigned maxAlign = 0;
      for (const mlir::Type elem : tupleType.getTypes()) {
         unsigned elemAlign = 0;
         size_t elemSize = 0;
         llvm::TypeSwitch<mlir::Type>(elem)
            .Case<dialect::util::RefType>([&](dialect::util::RefType t) {
               elemSize = sizeof(void*);
               elemAlign = alignof(void*);
            })
            .Case<mlir::IntegerType>([&](mlir::IntegerType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (elemSize) {
                  case 1: elemAlign = alignof(int8_t); break;
                  case 2: elemAlign = alignof(int16_t); break;
                  case 4: elemAlign = alignof(int32_t); break;
                  case 8: elemAlign = alignof(int64_t); break;
                  case 16: elemAlign = alignof(__int128); break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::FloatType>([&](mlir::FloatType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (t.getIntOrFloatBitWidth()) {
                  case 32:
                     static_assert(sizeof(float) == 4);
                     elemAlign = alignof(float);
                     break;
                  case 64:
                     static_assert(sizeof(double) == 8);
                     elemAlign = alignof(double);
                     break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::TupleType>([&](mlir::TupleType tupleType) {
               auto [size, align] = TupleHelper{tupleType}.sizeAndPadding();
               elemSize = size;
               elemAlign = align;
            })
            .Case<mlir::IndexType>([&](auto) {
               elemSize = sizeof(uint64_t);
               elemAlign = alignof(uint64_t);
            })
            .Default([](mlir::Type) {
               assert(false && "Cannot calculate size for unsupported type.");
               return 0;
            });
         // enforce alignment requirement
         offset += offset % elemAlign;
         // add offset of element
         offset += elemSize;
         maxAlign = std::max(maxAlign, elemAlign);
      }
      // adjust entire struct size to the maximum alignment (for array usage)
      offset += offset % maxAlign;
      return {offset, maxAlign};
   }
};
}
#endif //UTILS_HPP
