#ifndef LINGODB_EXECUTION_BASELINE_UTILS_HPP
#define LINGODB_EXECUTION_BASELINE_UTILS_HPP

#include <lingodb/compiler/Dialect/util/UtilTypes.h>

#include <llvm/ADT/TypeSwitch.h>

namespace lingodb::execution::baseline {
using namespace compiler;

class TupleHelper {
   public:
   TupleHelper() = default;

   [[nodiscard]] static std::pair<unsigned, size_t> getCxxAlignAndSize(const mlir::Type elem) noexcept {
      unsigned elemAlign = 0;
      size_t elemSize = 0;
      llvm::TypeSwitch<mlir::Type>(elem)
         .Case<dialect::util::RefType>([&](auto) {
            elemSize = sizeof(void*);
            elemAlign = alignof(void*);
         })
         .Case<mlir::IntegerType>([&](const mlir::IntegerType t) {
            elemSize = (t.getIntOrFloatBitWidth() + 7) / 8;
            switch (elemSize) {
               case 1:
                  elemAlign = alignof(int8_t);
                  break;
               case 2:
                  elemAlign = alignof(int16_t);
                  break;
               case 4:
                  elemAlign = alignof(int32_t);
                  break;
               case 8:
                  elemAlign = alignof(int64_t);
                  break;
               case 16:
                  elemAlign = alignof(__int128);
                  break;
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
         .Case<mlir::TupleType>([&](const mlir::TupleType tupleType) {
            auto [size, align] = sizeAndPadding(tupleType);
            elemSize = size;
            elemAlign = align;
         })
         .Case<mlir::IndexType>([&](auto) {
            elemSize = sizeof(uint64_t);
            elemAlign = alignof(uint64_t);
         })
         .Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
            elemSize = sizeof(__uint128_t);
            elemAlign = alignof(__uint128_t);
         })
         .Default([](mlir::Type) {
            assert(false && "Cannot calculate size for unsupported type.");
         });
      return {elemAlign, elemSize};
   }

   [[nodiscard]] static unsigned getElementOffset(const mlir::TupleType tupleType, const uint32_t index) noexcept {
      size_t offset = 0;
      assert(index < tupleType.getTypes().size() && "Index out of bounds for tuple type");
      for (size_t i = 0; i <= index; ++i) {
         const mlir::Type elem = tupleType.getTypes()[i];
         const auto [elemAlign, elemSize] = getCxxAlignAndSize(elem);
         // enforce alignment requirement
         if (offset % elemAlign != 0)
            offset += elemAlign - offset % elemAlign;
         if (i != index) {
            // add offset of element
            offset += elemSize;
         }
      }
      return offset;
   }

   [[nodiscard]] static std::pair<size_t, unsigned> sizeAndPadding(const mlir::TupleType tupleType) noexcept {
      size_t offset = 0;
      unsigned maxAlign = 0;
      for (const mlir::Type elem : tupleType.getTypes()) {
         const auto [elemAlign, elemSize] = getCxxAlignAndSize(elem);
         // enforce alignment requirement
         if (offset % elemAlign != 0)
            offset += elemAlign - offset % elemAlign;
         // add offset of element
         offset += elemSize;
         maxAlign = std::max(maxAlign, elemAlign);
      }
      if (offset == 0)
         return {0, 1}; // empty tuple case
      // adjust entire struct size to the maximum alignment (for array usage)
      if (offset % maxAlign != 0)
         offset += maxAlign - offset % maxAlign;
      return {offset, maxAlign};
   }

   [[nodiscard]] static size_t numSlots(const mlir::TupleType tupleType) noexcept {
      size_t count = 0;
      // TODO: make more efficient / not recursive!
      for (size_t i = 0; i < tupleType.getTypes().size(); ++i) {
         auto type = tupleType.getType(i);
         count += mlir::TypeSwitch<mlir::Type, uint32_t>(type)
                     .Case<mlir::IntegerType>([](auto intType) {
                        return (intType.getIntOrFloatBitWidth() + 64 - 1) / 64;
                     })
                     .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) { return 2; })
                     .template Case<mlir::TupleType>([](const mlir::TupleType t) { return TupleHelper::numSlots(t); })
                     .Default([](mlir::Type t) { return 1; });
      }
      return count;
   }

   [[nodiscard]] static mlir::Type typeAtSlot(const mlir::TupleType tupleType, const uint32_t pos) noexcept {
      size_t count = 0;
      for (const mlir::Type elem : tupleType.getTypes()) {
         const size_t elemCount = mlir::TypeSwitch<mlir::Type, size_t>(elem).Case<mlir::TupleType>([&](const mlir::TupleType t) {
                                                                               return numSlots(t);
                                                                            })
                                     .template Case<mlir::IntegerType>([](auto intType) {
                                        return (intType.getIntOrFloatBitWidth() + 64 - 1) / 64;
                                     })
                                     .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) { return 2; })
                                     .Default([&](mlir::Type) {
                                        return 1;
                                     });
         if (pos < count + elemCount) {
            // the requested position is within this element
            return mlir::TypeSwitch<mlir::Type, mlir::Type>(elem).Case<mlir::TupleType>([&](const mlir::TupleType t) {
                                                                    return typeAtSlot(t, pos - count);
                                                                 })
               .Default([&](const mlir::Type t) {
                  return t;
               });
         }
         count += elemCount;
      }
      assert(false && "Position out of bounds for tuple type");
      return nullptr;
   }
};
} // namespace lingodb::execution::baseline
#endif
