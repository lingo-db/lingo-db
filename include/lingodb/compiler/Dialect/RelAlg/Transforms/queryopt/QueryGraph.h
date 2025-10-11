#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPH_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPH_H

#include <lingodb/compiler/Dialect/DB/IR/DBOps.h>
#include <lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h>
#include <lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/utils.h>
#include <lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h>

#include "llvm/Support/Debug.h"

namespace lingodb::compiler::dialect::relalg {
class QueryGraph {
   public:
   size_t numNodes;
   size_t pseudoNodes = 0;
   NodeSet normalNodesMask;

   struct SelectionEdge {
      size_t id;
      llvm::DenseMap<NodeSet, double, HashNodeSet> cachedSel;
      NodeSet required;
      Operator op;
      double selectivity = 1;
      [[nodiscard]] bool connects(const NodeSet& s1, const NodeSet& s2) const {
         return required.isSubsetOf(s1 | s2) && !required.isSubsetOf(s1) && !required.isSubsetOf(s2);
      }
      [[nodiscard]] bool connects2(const NodeSet& complete, const NodeSet& s1, const NodeSet& s2) const {
         return required.isSubsetOf(complete) && !required.isSubsetOf(s1) && !required.isSubsetOf(s2);
      }
      std::pair<bool, size_t> findNeighbor(const NodeSet& s, const NodeSet& x) {
         auto remaining = required & ~s;
         if (!x.intersects(remaining) && remaining.any()) {
            return {true, remaining.findFirst()};
         }
         return {false, 0};
      }
   };
   struct JoinEdge {
      Operator op;
      NodeSet right;
      NodeSet left;
      double selectivity = 1;
      std::optional<size_t> createdNode;
      std::optional<std::pair<const lingodb::compiler::dialect::tuples::Column*, const lingodb::compiler::dialect::tuples::Column*>> equality;

      [[nodiscard]] bool connects(const NodeSet& s1, const NodeSet& s2) const {
         if (!(left.any() && right.any())) {
            return false;
         }
         if (left.isSubsetOf(s1) && right.isSubsetOf(s2)) {
            return true;
         }
         if (left.isSubsetOf(s2) && right.isSubsetOf(s1)) {
            return true;
         }
         return false;
      }

      std::pair<bool, size_t> findNeighbor(const NodeSet& s, const NodeSet& x) {
         if (left.isSubsetOf(s) && !s.intersects(right)) {
            auto otherside = right;
            if (!x.intersects(otherside) && otherside.any()) {
               return {true, otherside.findFirst()};
            }
         } else if (right.isSubsetOf(s) && !s.intersects(left)) {
            auto otherside = left;
            if (!x.intersects(otherside) && otherside.any()) {
               return {true, otherside.findFirst()};
            }
         }
         return {false, 0};
      }
   };

   struct Node {
      size_t id;
      Operator op;
      std::vector<Operator> additionalPredicates;
      double selectivity;
      double rows;

      explicit Node(Operator op) : op(op) {}

      std::vector<size_t> edges;
      std::vector<size_t> selections;
   };

   std::vector<Node> nodes;
   std::vector<JoinEdge> joins;
   std::vector<SelectionEdge> selections;
   llvm::DenseMap<size_t, size_t> pseudoNodeOwner;

   QueryGraph(size_t numNodes) : numNodes(numNodes) {}

   static void printReadable(const NodeSet& s, llvm::raw_ostream& out) {
      out << "{";
      for (auto pos : s) {
         out << pos << ",";
      }
      out << "}";
   }

   void print(llvm::raw_ostream& out);

   void dump() {
      print(llvm::dbgs());
   }
   size_t addPseudoNode() {
      pseudoNodes++;
      normalNodesMask = NodeSet::fillUntil(numNodes, numNodes - 1 - pseudoNodes);
      return numNodes - pseudoNodes;
   }
   static std::optional<std::pair<const lingodb::compiler::dialect::tuples::Column*, const lingodb::compiler::dialect::tuples::Column*>> analyzePredicate(PredicateOperator selection) {
      auto returnOp = mlir::cast<lingodb::compiler::dialect::tuples::ReturnOp>(selection.getPredicateBlock().getTerminator());
      if (returnOp.getResults().empty()) return {};
      mlir::Value v = returnOp.getResults()[0];
      if (auto cmpOp = mlir::dyn_cast_or_null<db::CmpOp>(v.getDefiningOp())) {
         if (!cmpOp.isEqualityPred(true)) return {};
         if (auto leftColref = mlir::dyn_cast_or_null<lingodb::compiler::dialect::tuples::GetColumnOp>(cmpOp.getLeft().getDefiningOp())) {
            if (auto rightColref = mlir::dyn_cast_or_null<lingodb::compiler::dialect::tuples::GetColumnOp>(cmpOp.getRight().getDefiningOp())) {
               return std::make_pair<const lingodb::compiler::dialect::tuples::Column*, const lingodb::compiler::dialect::tuples::Column*>(&leftColref.getAttr().getColumn(), &rightColref.getAttr().getColumn());
            }
         }
      }
      return {};
   }
   void addJoinEdge(NodeSet left, NodeSet right, Operator op, std::optional<size_t> createdNode) {
      assert(left.valid());
      assert(right.valid());

      size_t edgeid = joins.size();
      joins.emplace_back();
      JoinEdge& e = joins.back();
      if (op) {
         e.op = op;
         if (mlir::isa<lingodb::compiler::dialect::relalg::SelectionOp, lingodb::compiler::dialect::relalg::InnerJoinOp>(op)) {
            e.equality = analyzePredicate(mlir::cast<PredicateOperator>(e.op.getOperation()));
         }
      }
      e.left = left;
      e.right = right;
      e.createdNode = createdNode;
      if (createdNode) {
         pseudoNodeOwner[createdNode.value()] = edgeid;
         addJoinEdge(left | right, NodeSet::single(numNodes, createdNode.value()), {}, {});
      }
      for (auto n : left) {
         if (n >= nodes.size()) {
            //pseudo node
            continue;
         }
         nodes[n].edges.push_back(edgeid);
      }
      for (auto n : right) {
         if (n >= nodes.size()) {
            //pseudo node
            continue;
         }
         nodes[n].edges.push_back(edgeid);
      }
   }
   void addSelectionEdge(NodeSet required, Operator op) {
      if (required.count() == 2) {
         NodeSet left(numNodes);
         NodeSet right(numNodes);
         auto leftIdx = required.findFirst();
         auto rightIdx = required.findLast();
         if (leftIdx < nodes.size() && rightIdx < nodes.size()) {
            left.set(leftIdx);
            right.set(rightIdx);
            addJoinEdge(left, right, op, {});
            return;
         }
      }
      size_t edgeid = selections.size();
      selections.emplace_back();
      SelectionEdge& e = selections.back();
      if (op) {
         e.op = op;
      }
      e.required = required;
      e.id = edgeid;
      for (auto n : required) {
         if (n >= nodes.size()) {
            //pseudo node
            continue;
         }
         nodes[n].selections.push_back(edgeid);
      }
   }
   template<class Fn>
   void iterateNodesDesc(const Fn& fn) {
      for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
         fn(*it);
      }
   }
   template<class Fn>
   static void iterateSetDec(const NodeSet& s, const Fn& fn) {
      std::vector<size_t> positions;
      for (auto v : s) {
         positions.push_back(v);
      }
      for (auto it = positions.rbegin(); it != positions.rend(); it++) {
         fn(*it);
      }
   }
   bool containsLonelyPseudo(const NodeSet& s1) {
      if (s1.isSubsetOf(normalNodesMask)) { //fast lane: no pseudo nodes present
         return false;
      }
      for (auto pseudo : (s1 & ~normalNodesMask)) {
         auto join = joins[pseudoNodeOwner[pseudo]];
         if (join.left.isSubsetOf(s1) && join.right.isSubsetOf(s1)) {
            //pseudo but is not lonely
         } else {
            return true;
         }
      }
      return false;
   }
   bool isConnected(const NodeSet& s1, const NodeSet& s2) {
      if (containsLonelyPseudo(s1) || containsLonelyPseudo(s2)) {
         return false;
      }
      for (auto v : s1) {
         if (v >= nodes.size()) {
            //pseudo node
            continue;
         }
         Node& n = nodes[v];
         for (auto edgeid : n.edges) {
            auto& edge = joins[edgeid];
            if (edge.connects(s1, s2)) return true;
         }
         for (auto edgeid : n.selections) {
            auto& edge = selections[edgeid];
            if (edge.connects(s1, s2)) return true;
         }
      }
      return false;
   }
   [[nodiscard]] const std::vector<Node>& getNodes() const {
      return nodes;
   }
   std::vector<JoinEdge>& getJoinEdges() {
      return joins;
   }

   template<class Fn>
   void iterateNodes(NodeSet& s, const Fn& fn) {
      for (auto v : s) {
         if (v >= nodes.size()) {
            //pseudo node
            continue;
         }
         Node& n = nodes[v];
         fn(n);
      }
   }
   template<class Fn>
   void iterateJoinEdges(NodeSet& s, const Fn& fn) {
      iterateNodes(s, [&](Node& n) {
         for (auto edgeid : n.edges) {
            auto& edge = joins[edgeid];
            fn(edge);
         }
      });
   }
   template<class Fn>
   void iterateSelectionEdges(NodeSet& s, const Fn& fn) {
      iterateNodes(s, [&](Node& n) {
         for (auto edgeid : n.selections) {
            auto& edge = selections[edgeid];
            fn(edge);
         }
      });
   }

   NodeSet getNeighbors(NodeSet& s, NodeSet x) {
      NodeSet res(numNodes);
      iterateJoinEdges(s, [&](JoinEdge& edge) {
         auto [found, representative] = edge.findNeighbor(s, x);
         if (found) {
            res.set(representative);
         }
      });
      iterateSelectionEdges(s, [&](SelectionEdge& edge) {
         auto [found, representative] = edge.findNeighbor(s, x);
         if (found) {
            res.set(representative);
         }
      });
      return res;
   }
   struct Predicate {
      lingodb::compiler::dialect::relalg::ColumnSet left, right;
      bool isEq;
      Predicate(dialect::relalg::ColumnSet left, lingodb::compiler::dialect::relalg::ColumnSet right, bool isEq) : left(left), right(right), isEq(isEq) {}
   };
   std::vector<Predicate> analyzePred(mlir::Block* block, lingodb::compiler::dialect::relalg::ColumnSet availableLeft, lingodb::compiler::dialect::relalg::ColumnSet availableRight) {
      llvm::DenseMap<mlir::Value, lingodb::compiler::dialect::relalg::ColumnSet> required;
      std::vector<Predicate> predicates;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<lingodb::compiler::dialect::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), lingodb::compiler::dialect::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<lingodb::compiler::dialect::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred(true)) {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               if (leftAttributes.empty() || rightAttributes.empty()) return;
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  predicates.push_back(Predicate(leftAttributes, rightAttributes, true));
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  predicates.push_back(Predicate(rightAttributes, leftAttributes, true));
               }
            } else {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  predicates.push_back(Predicate(leftAttributes, rightAttributes, false));
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  predicates.push_back(Predicate(rightAttributes, leftAttributes, false));
               }
            }
         } else {
            lingodb::compiler::dialect::relalg::ColumnSet attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return predicates;
   }
   ColumnSet getAttributesForNodeSet(NodeSet& nodeSet) {
      ColumnSet a;
      iterateNodes(nodeSet, [&](Node& n) {
         a.insert(n.op.getAvailableColumns());
      });
      return a;
   }
   void addPredicates(std::vector<Predicate>& predicates, Operator op, lingodb::compiler::dialect::relalg::ColumnSet availableLeft, lingodb::compiler::dialect::relalg::ColumnSet availableRight) {
      if (auto predicateOp = mlir::dyn_cast_or_null<PredicateOperator>(op.getOperation())) {
         auto newPredicates = analyzePred(&predicateOp.getPredicateBlock(), availableLeft, availableRight);
         predicates.insert(predicates.end(), newPredicates.begin(), newPredicates.end());
      }
   }
   ColumnSet getPKey(dialect::relalg::QueryGraph::Node& n);
   double estimateSelectivity(Operator op, NodeSet left, NodeSet right);
   void estimate();
   double calculateSelectivity(SelectionEdge& edge, NodeSet left, NodeSet right);
};
} // namespace lingodb::compiler::dialect::relalg
#endif //LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPH_H
