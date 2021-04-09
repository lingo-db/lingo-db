#include "mlir/Dialect/RelAlg/queryopt/utils.h"
namespace mlir::relalg {
void node_set::iterateSubsets(const std::function<void(node_set)>& fn) const {
   if (!storage.any()) return;
   node_set s = *this;
   auto s1 = s & s.negate();
   while (s1 != s) {
      fn(s1);
      auto s1flipped = s1.flip();
      auto s2 = s & s1flipped;
      s1 = s & s2.negate();
   }
   fn(s);
}
node_set node_set::negate() const {
   node_set res = *this;
   size_t pos = res.find_first();
   size_t flipLen = res.storage.size() - pos - 1;
   if (flipLen) {
      llvm::SmallBitVector flipVector(res.storage.size());
      flipVector.set(pos + 1, res.storage.size());
      res.storage ^= flipVector;
   }
   return res;
}
} // namespace mlir::relalg