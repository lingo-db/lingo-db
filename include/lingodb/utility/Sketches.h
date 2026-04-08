
#ifndef LINGODB_UTILITY_SKETCHES_H
#define LINGODB_UTILITY_SKETCHES_H
#include <array>
#include <cstdint>
#include <limits>

namespace lingodb::utility {
class Serializer;
class Deserializer;
class HyperLogLogSketch {
   //number of hash bits used to determine the register
   static constexpr uint64_t p = 6;
   //number of registers
   static constexpr uint64_t m = 1 << p;
   //we are using 64-bit hash-values
   static constexpr uint64_t q = 64 - p;
   //registers (8bit are enough since registers[i]<=q+1=65<256 for all i)
   std::array<uint8_t, m> registers = {0};

   public:
   void add(uint64_t hash) {
      //determine register index
      uint64_t index = hash >> q;
      //determine the number of leading zeros
      uint8_t leadingZeros = __builtin_clzll((hash << p) | (1ull << (p - 1ull))) + 1ull;
      //update the register
      registers[index] = std::max(registers[index], leadingZeros);
   }
   void merge(const HyperLogLogSketch& other) {
      for (uint64_t i = 0; i < m; i++) {
         registers[i] = std::max(registers[i], other.registers[i]);
      }
   }
   double estimate();

   void serialize(Serializer& serializer) const;
   static HyperLogLogSketch deserialize(Deserializer& deserializer);
};
} // namespace lingodb::utility

#endif //LINGODB_UTILITY_SKETCHES_H