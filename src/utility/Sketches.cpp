#include "lingodb/utility/Sketches.h"

#include "lingodb/utility/Serialization.h"
#include <cmath>
using namespace lingodb::utility;

//functions from the paper
namespace {
double sigma(double x) {
   if (x == 1.0) {
      return std::numeric_limits<double>::infinity();
   }
   double zPrime;
   double y = 1.0;
   double z = x;
   do {
      x *= x;
      zPrime = z;
      z += x * y;
      y += y;
   } while (zPrime != z);
   return z;
}

double tau(double x) {
   if (x == 0.0 || x == 1.0) {
      return 0.0;
   }
   double zPrime;
   double y = 1.0;
   double z = 1.0 - x;
   do {
      x = std::sqrt(x);
      zPrime = z;
      y *= 0.5;
      z -= std::pow(1 - x, 2) * y;
   } while (zPrime != z);
   return z / 3;
}

} //namespace
double HyperLogLogSketch::estimate() {
   //first compute the multiplicity vector
   std::array<uint32_t, q + 2> c = {0};
   for (uint64_t i = 0; i < m; i++) {
      c[registers[i]]++;
   }
   //compute the estimate
   double z = m * tau(static_cast<double>(m - c[q + 1]) / static_cast<double>(m));
   for (int k = q; k >= 1; k--) {
      z += c[k];
      z *= 0.5;
   }
   z += m * sigma(static_cast<double>(c[0]) / static_cast<double>(m));
   return (static_cast<double>(m) * m / (2.0 * std::log(2))) / z;
}
void HyperLogLogSketch::serialize(lingodb::utility::Serializer& serializer) const {
   for (uint64_t i = 0; i < m; i++) {
      serializer.writeProperty(i, registers[i]);
   }
}
HyperLogLogSketch HyperLogLogSketch::deserialize(lingodb::utility::Deserializer& deserializer) {
   HyperLogLogSketch hll;
   for (uint64_t i = 0; i < m; i++) {
      hll.registers[i] = deserializer.readProperty<uint8_t>(i);
   }
   return hll;
}