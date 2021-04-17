#ifndef DB_DIALECTS_UTILS_H
#define DB_DIALECTS_UTILS_H
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <mlir/Dialect/RelAlg/IR/RelAlgOps.h>

namespace mlir::relalg {
class node_set {
   public:
   llvm::SmallBitVector storage;

   public:
   node_set() = default;
   explicit node_set(size_t size) : storage(size) {}
   [[nodiscard]] node_set negate() const;
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
   void iterateSubsets(const std::function<void(node_set)>& fn) const;
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
class Plan {
   Operator op;
   std::vector<std::shared_ptr<Plan>> subplans;
   std::vector<Operator> additional_ops;
   size_t cost;
   std::string description;
   std::string dumpNode();
   Operator realizePlanRec();

   public:
   Plan(Operator op, const std::vector<std::shared_ptr<Plan>>& subplans, const std::vector<Operator>& additional_ops, size_t cost) : op(op), subplans(subplans), additional_ops(additional_ops), cost(cost) {}
   Operator realizePlan();
   void dump();
   size_t getCost() const;
   void setDescription(const std::string& descr);
   const std::string& getDescription() const;
};
}
#endif //DB_DIALECTS_UTILS_H
