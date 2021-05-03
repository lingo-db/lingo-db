#ifndef MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPH_H
#define MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPH_H

#include "llvm/Support/Debug.h"
#include <mlir/Dialect/RelAlg/IR/RelAlgOps.h>
#include <mlir/Dialect/RelAlg/queryopt/utils.h>
namespace mlir::relalg {
class QueryGraph {
   public:
   size_t numNodes;
   enum class EdgeType {
      REAL,
      IMPLICIT,
      IGNORE
   };
   struct Edge {
      Operator op;
      EdgeType edgeType = EdgeType::REAL;
      NodeSet right;
      NodeSet left;
      NodeSet arbitrary;
      [[nodiscard]] bool connects(const NodeSet& s1, const NodeSet& s2) const {
         if (!((left.any() && right.any()) || (left.any() && arbitrary.any()) || (arbitrary.any() && right.any()))) {
            return false;
         }
         if (left.isSubsetOf(s1) && right.isSubsetOf(s2) && arbitrary.isSubsetOf(s1 | s2)) {
            return true;
         }
         if (left.isSubsetOf(s2) && right.isSubsetOf(s1) && arbitrary.isSubsetOf(s1 | s2)) {
            return true;
         }
         return false;
      }

      std::pair<bool, size_t> findNeighbor(const NodeSet& s, const NodeSet& x) {
         if (left.isSubsetOf(s) && !s.intersects(right)) {
            auto otherside = right | (arbitrary & ~s);
            if (!x.intersects(otherside) && otherside.any()) {
               return {true, otherside.findFirst()};
            }
         } else if (right.isSubsetOf(s) && !s.intersects(left)) {
            auto otherside = left | (arbitrary & ~s);
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
      NodeSet dependencies;

      explicit Node(Operator op) : op(op) {}

      std::vector<size_t> edges;
   };

   std::vector<Node> nodes;
   std::vector<Edge> edges;

   QueryGraph(size_t numNodes) : numNodes(numNodes) {}

   static void printReadable(const NodeSet& s, llvm::raw_ostream& out) {
      out << "{";
      for (auto s : s) {
         out << s << ",";
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
      out << "Edges: [\n";
      for (auto& e : edges) {
         out << "{ v=";
         printReadable(e.left, out);
         out << ", u=";
         printReadable(e.right, out);
         out << ", w=";
         printReadable(e.arbitrary, out);
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

   void addEdge(NodeSet left, NodeSet right, NodeSet arbitrary, Operator op, EdgeType edgeType) {
      assert(left.valid());
      assert(right.valid());
      assert(arbitrary.valid());

      size_t edgeid = edges.size();
      edges.emplace_back();
      Edge& e = edges.back();
      if (op) {
         e.op = op;
      }
      e.edgeType = edgeType;
      e.left = left;
      e.right = right;
      e.arbitrary = arbitrary;
      for (auto n : left) {
         nodes[n].edges.push_back(edgeid);
      }
      for (auto n : right) {
         nodes[n].edges.push_back(edgeid);
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

   bool isConnected(const NodeSet& s1, const NodeSet& s2) {
      bool found = false;
      for (auto v : s1) {
         Node& n = nodes[v];
         for (auto edgeid : n.edges) {
            auto& edge = edges[edgeid];
            found |= edge.connects(s1, s2);
         }
      }
      return found;
   }
   [[nodiscard]] const std::vector<Node>& getNodes() const {
      return nodes;
   }
   std::vector<Edge>& getEdges() {
      return edges;
   }
   void iterateNodes(NodeSet& s, const std::function<void(Node&)>& fn) {
      for (auto v : s) {
         Node& n = nodes[v];
         fn(n);
      }
   }

   void iterateEdges(NodeSet& s, const std::function<void(Edge&)>& fn) {
      iterateNodes(s, [&](Node& n) {
         for (auto edgeid : n.edges) {
            auto& edge = edges[edgeid];
            fn(edge);
         }
      });
   }

   NodeSet getNeighbors(NodeSet& s, NodeSet x) {
      NodeSet res(numNodes);
      iterateEdges(s, [&](Edge& edge) {
         auto [found, representative] = edge.findNeighbor(s, x);
         if (found) {
            res.set(representative);
         }
      });
      return res;
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPH_H
