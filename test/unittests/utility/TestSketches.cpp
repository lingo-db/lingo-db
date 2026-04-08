
#include "catch2/catch_all.hpp"

#include "lingodb/utility/Serialization.h"
#include "lingodb/utility/Sketches.h"

using namespace lingodb::utility;

namespace {
uint64_t murmur64(uint64_t h) {
   h ^= h >> 33;
   h *= 0xff51afd7ed558ccdull;
   h ^= h >> 33;
   h *= 0xc4ceb9fe1a85ec53ull;
   h ^= h >> 33;
   return h;
}
} // namespace

TEST_CASE("HyperLogLog:Basic") {
   HyperLogLogSketch hll;
   for (uint64_t i = 1; i <= 500; i++) {
      hll.add(murmur64(i));
   }
   for (uint64_t i = 200; i <= 1000; i++) {
      hll.add(murmur64(i));
   }
   auto estimate = hll.estimate();
   REQUIRE((estimate > 900 && estimate < 1100));
}
TEST_CASE("HyperLogLog:Merge") {
   HyperLogLogSketch hll1;
   for (uint64_t i = 1; i <= 500; i++) {
      hll1.add(murmur64(i));
   }
   HyperLogLogSketch hll2;
   for (uint64_t i = 200; i <= 1000; i++) {
      hll2.add(murmur64(i));
   }
   hll1.merge(hll2);
   auto estimate = hll1.estimate();
   REQUIRE((estimate > 900 && estimate < 1100));
}
TEST_CASE("HyperLogLog: Serialization") {
   HyperLogLogSketch hll;
   for (uint64_t i = 1; i <= 500; i++) {
      hll.add(murmur64(i));
   }
   for (uint64_t i = 200; i <= 1000; i++) {
      hll.add(murmur64(i));
   }
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, hll);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   HyperLogLogSketch hll2 = deserializer.readProperty<HyperLogLogSketch>(1);
   auto estimate = hll2.estimate();
   REQUIRE((estimate > 900 && estimate < 1100));
}