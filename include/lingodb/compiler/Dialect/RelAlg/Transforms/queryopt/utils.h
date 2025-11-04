#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_UTILS_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_UTILS_H

#include "lingodb/compiler/mlir-support/eval.h"

#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h>
namespace lingodb::compiler::dialect::relalg {
class QueryGraph;
class NodeSet {
   public:
   llvm::SmallBitVector storage;

   public:
   NodeSet() = default;
   explicit NodeSet(const llvm::SmallBitVector& storage) : storage(storage) {}
   explicit NodeSet(size_t size) : storage(size) {}
   [[nodiscard]] NodeSet negate() const {
      NodeSet res = *this;
      size_t pos = res.findFirst();
      size_t flipLen = res.storage.size() - pos - 1;
      if (flipLen) {
         llvm::SmallBitVector flipVector(res.storage.size());
         flipVector.set(pos + 1, res.storage.size());
         res.storage ^= flipVector;
      }
      return res;
   }
   static NodeSet ones(size_t size) {
      NodeSet res(size);
      res.storage.set();
      return res;
   }
   static NodeSet fillUntil(size_t numNodes, size_t n) {
      auto res = NodeSet(numNodes);
      res.storage.set(0, n + 1);
      return res;
   }

   static NodeSet single(size_t numNodes, size_t pos) {
      auto res = NodeSet(numNodes);
      res.set(pos);
      return res;
   }
   [[nodiscard]] bool isSubsetOf(const NodeSet& s) const {
      return !storage.test(s.storage);
   }
   [[nodiscard]] bool intersects(const NodeSet& rhs) const {
      return (storage & rhs.storage).any();
   }
   void set(size_t pos) {
      assert(pos < storage.size());
      storage.set(pos);
   }
   [[nodiscard]] auto begin() const {
      return storage.set_bits_begin();
   }
   [[nodiscard]] auto end() const {
      return storage.set_bits_end();
   }
   [[nodiscard]] size_t findFirst() const {
      return storage.find_first();
   }
   [[nodiscard]] size_t findLast() const {
      return storage.find_last();
   }
   bool operator==(const NodeSet& rhs) const { return storage == rhs.storage; }
   bool operator!=(const NodeSet& rhs) const { return storage != rhs.storage; }
   bool operator<(const NodeSet& rhs) const {
      int diff = (storage ^ rhs.storage).find_last();
      return diff >= 0 ? rhs.storage.test(diff) : false;
   }

   [[nodiscard]] bool any() const {
      return storage.any();
   }
   NodeSet& operator|=(const NodeSet& rhs) {
      storage |= rhs.storage;
      return *this;
   }
   NodeSet& operator&=(const NodeSet& rhs) {
      storage &= rhs.storage;
      return *this;
   }
   NodeSet operator&(
      const NodeSet& rhs) const {
      NodeSet result = *this;
      result &= rhs;
      return result;
   }
   NodeSet operator~() const {
      NodeSet res = flip();
      return res;
   }
   NodeSet operator|(const NodeSet& rhs) const {
      NodeSet result = *this;
      result |= rhs;
      return result;
   }
   [[nodiscard]] bool test(size_t pos) const {
      return storage.test(pos);
   }
   [[nodiscard]] bool valid() const {
      return !storage.empty();
   }
   [[nodiscard]] size_t count() const {
      return storage.count();
   }
   [[nodiscard]] NodeSet flip() const {
      NodeSet res = *this;
      res.storage.flip();
      return res;
   }
   template <class Fn>
   void iterateSubsets(const Fn& fn) const {
      if (!storage.any()) return;
      NodeSet s = *this;
      auto s1 = s & s.negate();
      while (s1 != s) {
         fn(s1);
         auto s1flipped = s1.flip();
         auto s2 = s & s1flipped;
         s1 = s & s2.negate();
      }
      fn(s);
   }
   [[nodiscard]] size_t hash() const {
      return llvm::DenseMapInfo<llvm::SmallBitVector>::getHashValue(storage);
   }
   [[nodiscard]] size_t size() const {
      return storage.size();
   }
};
struct HashNodeSet {
   static bool isEqual(const NodeSet& lhs, const NodeSet& rhs) {
      return llvm::DenseMapInfo<llvm::SmallBitVector>::isEqual(lhs.storage, rhs.storage);
   }
   static unsigned getHashValue(const NodeSet& s) {
      return llvm::DenseMapInfo<llvm::SmallBitVector>::getHashValue(s.storage);
   }
   static NodeSet getEmptyKey() {
      return NodeSet(llvm::DenseMapInfo<llvm::SmallBitVector>::getEmptyKey());
   }
   static NodeSet getTombstoneKey() {
      return NodeSet(llvm::DenseMapInfo<llvm::SmallBitVector>::getTombstoneKey());
   }
};
class Plan {
   Operator op;
   std::vector<std::shared_ptr<Plan>> subplans;
   std::vector<Operator> additionalOps;
   double cost;
   double rows;
   std::string description;
   std::string dumpNode();
   Operator realizePlanRec();

   public:
   Plan(Operator op, const std::vector<std::shared_ptr<Plan>>& subplans, const std::vector<Operator>& additionalOps, double rows) : op(op), subplans(subplans), additionalOps(additionalOps), cost(rows), rows(rows) {
      for (auto subplan : subplans) {
         cost += subplan->getCost();
      }
   }
   Operator realizePlan();
   void dump();
   double getCost() const;
   double getRows() const {
      return rows;
   }
   void setDescription(const std::string& descr);
   const std::string& getDescription() const;

   static std::shared_ptr<Plan> joinPlans(NodeSet leftProblem, NodeSet rightProblem, std::shared_ptr<Plan> leftPlan, std::shared_ptr<Plan> rightPlan, QueryGraph& queryPlan, NodeSet& combinedProblem);
};

std::unique_ptr<support::eval::expr> buildEvalExpr(mlir::Value val, llvm::DenseMap<const lingodb::compiler::dialect::tuples::Column*, std::string>& mapping);
} // namespace lingodb::compiler::dialect::relalg
#endif //LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_UTILS_H
