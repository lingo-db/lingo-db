#ifndef DB_DIALECTS_UTILS_H
#define DB_DIALECTS_UTILS_H
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/TypeSwitch.h>
#include <cstddef>
#include <iterator>

namespace mlir::relalg {
class node_set {
   public:
   llvm::SmallBitVector storage;

   public:
   node_set() = default;
   explicit node_set(size_t size) : storage(size) {}
   [[nodiscard]] node_set negate() const {
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
   static node_set ones(size_t size) {
      node_set res(size);
      res.storage.set();
      return res;
   }
   static node_set fill_until(size_t num_nodes, size_t n) {
      auto res = node_set(num_nodes);
      res.storage.set(0, n + 1);
      return res;
   }

   static node_set single(size_t num_nodes, size_t pos) {
      auto res = node_set(num_nodes);
      res.set(pos);
      return res;
   }
   [[nodiscard]] bool is_subset_of(const node_set& S) const {
      return (storage & S.storage) == storage;
   }
   [[nodiscard]] bool intersects(const node_set& rhs) const {
      return (storage & rhs.storage).any();
   }
   void set(size_t pos) {
      storage.set(pos);
   }
   [[nodiscard]] auto begin() const {
      return storage.set_bits_begin();
   }
   [[nodiscard]] auto end() const {
      return storage.set_bits_end();
   }
   [[nodiscard]] size_t find_first() const {
      return storage.find_first();
   }
   bool operator==(const node_set& rhs) const { return storage == rhs.storage; }
   bool operator!=(const node_set& rhs) const { return storage != rhs.storage; }
   bool operator<(const node_set& rhs) const {
      int diff = (storage ^ rhs.storage).find_last();
      return diff >= 0 ? rhs.storage.test(diff) : false;
   }

   [[nodiscard]] bool any() const {
      return storage.any();
   }
   node_set& operator|=(const node_set& rhs) {
      storage |= rhs.storage;
      return *this;
   }
   node_set& operator&=(const node_set& rhs) {
      storage &= rhs.storage;
      return *this;
   }
   node_set operator&(
      const node_set& rhs) const {
      node_set result = *this;
      result &= rhs;
      return result;
   }
   node_set operator~() const {
      node_set res = flip();
      return res;
   }
   node_set operator|(const node_set& rhs) const {
      node_set result = *this;
      result |= rhs;
      return result;
   }
   [[nodiscard]] bool valid() const {
      return !storage.empty();
   }
   [[nodiscard]] size_t count() const {
      return storage.count();
   }
   [[nodiscard]] node_set flip() const {
      node_set res = *this;
      res.storage.flip();
      return res;
   }
   void iterateSubsets(const std::function<void(node_set)>& fn) const {
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
   [[nodiscard]] size_t hash() const {
      return llvm::DenseMapInfo<llvm::SmallBitVector>::getHashValue(storage);
   }
   [[nodiscard]] size_t size() const {
      return storage.size();
   }
};
struct hash_node_set {
   size_t operator()(const node_set& bitset) const {
      return bitset.hash();
   }
};

}
#endif //DB_DIALECTS_UTILS_H
