#ifndef MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPH_H
#define MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPH_H

#include "llvm/Support/Debug.h"
#include <mlir/Dialect/DB/IR/DBOps.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgOps.h>
#include <mlir/Dialect/RelAlg/Transforms/queryopt/utils.h>
namespace mlir::relalg {
class QueryGraph {
   public:
   size_t numNodes;
   size_t pseudoNodes = 0;

   struct SelectionEdge {
      size_t id;
      std::unordered_map<NodeSet,double,HashNodeSet> cachedSel;
      NodeSet required;
      Operator op;
      [[nodiscard]] bool connects(const NodeSet& s1, const NodeSet& s2) const {
         return required.isSubsetOf(s1 | s2) && !required.isSubsetOf(s1) && !required.isSubsetOf(s2);
      }
      [[nodiscard]] bool connects2(const NodeSet& complete,const NodeSet& s1, const NodeSet& s2) const {
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
      llvm::Optional<size_t> createdNode;
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
   std::unordered_map<size_t, size_t> pseudoNodeOwner;

   QueryGraph(size_t numNodes) : numNodes(numNodes) {}

   static void printReadable(const NodeSet& s, llvm::raw_ostream& out) {
      out << "{";
      for (auto pos : s) {
         out << pos << ",";
      }
      out << "}";
   }

   void print(llvm::raw_ostream& out) {
      out << "QueryGraph:{\n";
      out << "Nodes: [\n";
      for (auto& n : nodes) {
         out << "{" << n.id << ",";
         n.op->print(out);
         out << ", predicates={";
         for (auto op : n.additionalPredicates) {
            op->print(out);
            out << ",";
         }

         out << "}}";
         out << "},\n";
      }
      out << "]\n";
      out << "Joins: [\n";
      for (auto& e : joins) {
         out << "{ v=";
         printReadable(e.left, out);
         out << ", u=";
         printReadable(e.right, out);
         out << ", op=\n";
         if (e.op) {
            e.op->print(out);
         }
         out << "}";
         out << "},\n";
      }
      out << "]\n";
      out << "Selections: [\n";
      for (auto& e : selections) {
         out << "{ required=";
         printReadable(e.required, out);
         out << ", op=\n";
         if (e.op) {
            e.op->print(out);
         }
         out << "}";
         out << "},\n";
      }
      out << "]\n";
      out << "}\n";
   }

   void dump() {
      print(llvm::dbgs());
   }
   size_t addPseudoNode() {
      pseudoNodes++;
      return numNodes - pseudoNodes;
   }
   void addJoinEdge(NodeSet left, NodeSet right, Operator op, llvm::Optional<size_t> createdNode) {
      assert(left.valid());
      assert(right.valid());

      size_t edgeid = joins.size();
      joins.emplace_back();
      JoinEdge& e = joins.back();
      if (op) {
         e.op = op;
      }
      e.left = left;
      e.right = right;
      e.createdNode = createdNode;
      if (createdNode) {
         pseudoNodeOwner[createdNode.getValue()] = edgeid;
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
      size_t edgeid = selections.size();
      selections.emplace_back();
      SelectionEdge& e = selections.back();
      if (op) {
         e.op = op;
      }
      e.required = required;
      e.id=edgeid;
      for (auto n : required) {
         if (n >= nodes.size()) {
            //pseudo node
            continue;
         }
         nodes[n].selections.push_back(edgeid);
      }
   }

   void iterateNodesDesc(const std::function<void(Node&)>& fn) {
      for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
         fn(*it);
      }
   }

   static void iterateSetDec(const NodeSet& s, const std::function<void(size_t)>& fn) {
      std::vector<size_t> positions;
      for (auto v : s) {
         positions.push_back(v);
      }
      for (auto it = positions.rbegin(); it != positions.rend(); it++) {
         fn(*it);
      }
   }
   bool containsLonelyPseudo(const NodeSet& s1) {
      auto mask = NodeSet::fillUntil(numNodes, numNodes - 1 - pseudoNodes);
      if ((s1 & mask) == s1) { //fast lane: no pseudo nodes present
         return false;
      }
      for (auto pseudo : (s1 & ~mask)) {
         if ((s1 & NodeSet::single(numNodes, pseudoNodeOwner[pseudo])).any()) {
            //pseudo but is not lonley
         } else {
            return true;
         }
      }
      return false;
   }
   bool isConnected(const NodeSet& s1, const NodeSet& s2) {
      bool found = false;
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
            found |= edge.connects(s1, s2);
         }
         for (auto edgeid : n.selections) {
            auto& edge = selections[edgeid];
            found |= edge.connects(s1, s2);
         }
      }
      return found;
   }
   [[nodiscard]] const std::vector<Node>& getNodes() const {
      return nodes;
   }
   std::vector<JoinEdge>& getJoinEdges() {
      return joins;
   }

   void iterateNodes(NodeSet& s, const std::function<void(Node&)>& fn) {
      for (auto v : s) {
         if (v >= nodes.size()) {
            //pseudo node
            continue;
         }
         Node& n = nodes[v];
         fn(n);
      }
   }

   void iterateJoinEdges(NodeSet& s, const std::function<void(JoinEdge&)>& fn) {
      iterateNodes(s, [&](Node& n) {
         for (auto edgeid : n.edges) {
            auto& edge = joins[edgeid];
            fn(edge);
         }
      });
   }
   void iterateSelectionEdges(NodeSet& s, const std::function<void(SelectionEdge&)>& fn) {
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
      mlir::relalg::Attributes left, right;
      bool isEq;
      Predicate(mlir::relalg::Attributes left, mlir::relalg::Attributes right, bool isEq) : left(left), right(right), isEq(isEq) {}
   };
   std::vector<Predicate> analyzePred(mlir::Block* block, mlir::relalg::Attributes availableLeft, mlir::relalg::Attributes availableRight) {
      llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required;
      std::vector<Predicate> predicates;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::Attributes::from(getAttr.attr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(op)) {
            if (cmpOp.predicate() == mlir::db::DBCmpPredicate::eq) {
               auto leftAttributes = required[cmpOp.left()];
               auto rightAttributes = required[cmpOp.right()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  predicates.push_back(Predicate(leftAttributes, rightAttributes, true));
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  predicates.push_back(Predicate(rightAttributes, leftAttributes, true));
               }
            } else {
               auto leftAttributes = required[cmpOp.left()];
               auto rightAttributes = required[cmpOp.right()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  predicates.push_back(Predicate(leftAttributes, rightAttributes, false));
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  predicates.push_back(Predicate(rightAttributes, leftAttributes, false));
               }
            }
         } else {
            mlir::relalg::Attributes attributes;
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
   Attributes getAttributesForNodeSet(NodeSet& nodeSet) {
      Attributes a;
      iterateNodes(nodeSet, [&](Node& n) {
         a.insert(n.op.getAvailableAttributes());
      });
      return a;
   }
   void addPredicates(std::vector<Predicate>& predicates, Operator op, mlir::relalg::Attributes availableLeft, mlir::relalg::Attributes availableRight) {
      if (auto predicateOp = mlir::dyn_cast_or_null<PredicateOperator>(op.getOperation())) {
         auto newPredicates = analyzePred(&predicateOp.getPredicateBlock(), availableLeft, availableRight);
         predicates.insert(predicates.end(), newPredicates.begin(), newPredicates.end());
      }
   }
   Attributes getPKey(StringRef scope, mlir::Attribute pkeyAttribute) {
      Attributes res;
      if (auto arrAttr = pkeyAttribute.dyn_cast_or_null<mlir::ArrayAttr>()) {
         for (auto attr : arrAttr) {
            if (auto stringAttr = attr.dyn_cast_or_null<mlir::StringAttr>()) {
               auto relAttrManager = pkeyAttribute.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
               res.insert(relAttrManager.get(scope, stringAttr.getValue()).get());
            }
         }
      }
      return res;
   }
   double estimateSelectivity(Operator op,NodeSet left,NodeSet right){
      auto availableLeft = getAttributesForNodeSet(left);
      auto availableRight = getAttributesForNodeSet(right);
      std::vector<Predicate> predicates;
      addPredicates(predicates, op, availableLeft, availableRight);
      double selectivity = 1.0;
      std::vector<std::pair<double, Attributes>> pkeysLeft;
      std::vector<std::pair<double, Attributes>> pkeysRight;
      iterateNodes(left, [&](auto node) {
         if (node.op) {
            if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())) {
               pkeysLeft.push_back({node.rows, getPKey(baseTableOp.sym_name(), baseTableOp->getAttr("pkey"))});
            }
         }
      });
      iterateNodes(right, [&](auto node) {
         if (node.op) {
            if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())) {
               pkeysRight.push_back({node.rows, getPKey(baseTableOp.sym_name(), baseTableOp->getAttr("pkey"))});
            }
         }
      });

      Attributes predicatesLeft;
      Attributes predicatesRight;
      for (auto predicate : predicates) {
         predicatesLeft.insert(predicate.left);
         predicatesRight.insert(predicate.right);
      }
      for (auto p : pkeysLeft) {
         auto [rows, pkey] = p;
         if (pkey.isSubsetOf(predicatesLeft)) {
            selectivity *= 1 / rows;
            predicatesLeft.remove(pkey);
         }
      }
      for (auto p : pkeysRight) {
         auto [rows, pkey] = p;
         if (pkey.isSubsetOf(predicatesRight)) {
            selectivity *= 1 / rows;
            predicatesRight.remove(pkey);
         }
      }
      for (auto predicate : predicates) {
         if (predicate.left.isSubsetOf(predicatesLeft) && predicate.right.isSubsetOf(predicatesRight)) {
            if (predicate.isEq) {
               selectivity *= 0.1;
            } else {
               selectivity *= 0.25;
            }
         }
      }
      return selectivity;
   }
   void estimate() {
      for (auto& node : nodes) {
         node.selectivity = 1;
         if (node.op) {
            node.rows = 1;
            if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())) {
               if (baseTableOp->hasAttr("rows")) {
                  node.rows = baseTableOp->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>().getInt();
               }
            }
            auto availableLeft = node.op.getAvailableAttributes();
            mlir::relalg::Attributes availableRight;
            std::vector<Predicate> predicates;
            for (auto pred : node.additionalPredicates) {
               addPredicates(predicates, pred, availableLeft, availableRight);
            }
            Attributes pkey;
            if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())) {
               pkey = getPKey(baseTableOp.sym_name(), baseTableOp->getAttr("pkey"));
            }
            Attributes predicatesLeft;
            for (auto predicate : predicates) {
               predicatesLeft.insert(predicate.left);
            }
            bool pKeyIncluded = pkey.isSubsetOf(predicatesLeft);
            if (pKeyIncluded) {
               node.selectivity = 1 / node.rows;
            } else {
               for (auto predicate : predicates) {
                  if (predicate.isEq) {
                     node.selectivity *= 0.1;
                  } else {
                     node.selectivity *= 0.25;
                  }
               }
            }
         }
      }
      for (auto& edge : joins) {
         edge.selectivity= estimateSelectivity(edge.op,edge.left,edge.right);
      }
   }
   double calculateSelectivity(SelectionEdge& edge,NodeSet left,NodeSet right) {
      auto key=left&edge.required;
      if(edge.cachedSel.contains(key)){
         return edge.cachedSel[key];
      }
      double selectivity= estimateSelectivity(edge.op,left,right);
      edge.cachedSel[key]=selectivity;
      return selectivity;
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPH_H
