#include "mlir/Dialect/RelAlg/queryopt/utils.h"
namespace mlir::relalg {
void node_set::iterateSubsets(const std::function<void(node_set)>& fn) const {
   if (!storage.any()) return;
   node_set S = *this;
   auto S1 = S & S.negate();
   while (S1 != S) {
      fn(S1);
      auto S1flipped = S1.flip();
      auto S2 = S & S1flipped;
      S1 = S & S2.negate();
   }
   fn(S);
}
node_set node_set::negate() const {
   node_set res = *this;
   size_t pos = res.find_first();
   size_t flip_len = res.storage.size() - pos - 1;
   if (flip_len) {
      llvm::SmallBitVector flip_vector(res.storage.size());
      flip_vector.set(pos + 1, res.storage.size());
      res.storage ^= flip_vector;
   }
   return res;
}
}