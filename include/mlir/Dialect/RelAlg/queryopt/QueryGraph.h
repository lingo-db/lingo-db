#ifndef DB_DIALECTS_QUERYGRAPH_H
#define DB_DIALECTS_QUERYGRAPH_H

#include "dynamic_bitset.h"
#include "llvm/Support/Debug.h"
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/TypeSwitch.h>

#include <functional>
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>
#include <mlir/Dialect/RelAlg/IR/RelAlgOps.h>
#include <mlir/Dialect/RelAlg/queryopt/utils.h>
namespace mlir::relalg {
class QueryGraph {
   public:
   size_t num_nodes;
   enum class EdgeType {
      REAL,
      IMPLICIT,
      IGNORE
   };
   struct Edge {
      Operator op;
      EdgeType edgeType;
      node_set right;
      node_set left;
      node_set arbitrary;
      bool connects(node_set S1, node_set S2) {
         if (!((left.any() && right.any()) || (left.any() && arbitrary.any()) || (arbitrary.any() && right.any()))) {
            return false;
         }
         if (left.is_subset_of(S1) && right.is_subset_of(S2) && arbitrary.is_subset_of(S1 | S2)) {
            return true;
         }
         if (left.is_subset_of(S2) && right.is_subset_of(S1) && arbitrary.is_subset_of(S1 | S2)) {
            return true;
         }
         return false;
      }

      std::pair<bool, size_t> findNeighbor(node_set S, node_set X) {
         if (left.is_subset_of(S) && !S.intersects(right)) {
            auto otherside = right | (arbitrary & ~S);
            if (!X.intersects(otherside) && otherside.any()) {
               return {true, otherside.find_first()};
            }
         } else if (right.is_subset_of(S) && !S.intersects(left)) {
            auto otherside = left | (arbitrary & ~S);
            if (!X.intersects(otherside) && otherside.any()) {
               return {true, otherside.find_first()};
            }
         }
         return {false, 0};
      }
   };

   struct Node {
      size_t id;
      Operator op;
      std::vector<Operator> additional_predicates;
      size_t cardinality;
      node_set dependencies;

      Node(Operator op) : op(op) {}

      std::vector<size_t> edges;
   };

   std::vector<Node> nodes;
   std::vector<Edge> edges;

   QueryGraph(size_t num_nodes, std::unordered_set<mlir::Operation*>& already_optimized) : num_nodes(num_nodes) {}

   void print_readable(node_set S, llvm::raw_ostream& out) {
      out << "{";
      iterateNodes(S, [&](Node& n) { out << n.id << ","; });
      out << "}";
   }

   void print(llvm::raw_ostream& out) {
      out << "QueryGraph:{\n";
      out << "Nodes: [\n";
      for (auto& n : nodes) {
         out << "{" << n.id << ",";
         n.op->print(out);
         out << ", predicates={";
         for (auto op : n.additional_predicates) {
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
         print_readable(e.left, out);
         out << ", u=";
         print_readable(e.right, out);
         out << ", w=";
         print_readable(e.arbitrary, out);
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

   void addEdge(node_set left, node_set right, node_set arbitrary, Operator op, EdgeType edgeType) {
      assert(left.valid());
      assert(right.valid());
      assert(arbitrary.valid());

      size_t edgeid = edges.size();
      edges.push_back(Edge());
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

   void iterateNodes(std::function<void(Node&)> fn) {
      iterateNodesDesc(fn);
   }

   void iterateNodesDesc(std::function<void(Node&)> fn) {
      for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
         fn(*it);
      }
   }

   void iterateSetDec(node_set S, std::function<void(size_t)> fn) {
      std::vector<size_t> positions;
      for (auto v : S) {
         positions.push_back(v);
      }
      for (auto it = positions.rbegin(); it != positions.rend(); it++) {
         fn(*it);
      }
   }

   bool isConnected(node_set S1, node_set S2) {
      bool found = false;
      for (auto v : S1) {
         Node& n = nodes[v];
         for (auto edgeid : n.edges) {
            auto& edge = edges[edgeid];
            found |= edge.connects(S1, S2);
         }
      }
      return found;
   }
   const std::vector<Node>& getNodes() const {
      return nodes;
   }
   std::vector<Edge>& getEdges() {
      return edges;
   }
   void iterateNodes(node_set S, std::function<void(Node&)> fn) {
      for (auto v : S) {
         Node& n = nodes[v];
         fn(n);
      }
   }

   void iterateEdges(node_set S, std::function<void(Edge&)> fn) {
      iterateNodes(S, [&](Node& n) {
         for (auto edgeid : n.edges) {
            auto& edge = edges[edgeid];
            fn(edge);
         }
      });
   }

   node_set getNeighbors(node_set S, node_set X) {
      node_set res(num_nodes);
      iterateEdges(S, [&](Edge& edge) {
         auto [found, representative] = edge.findNeighbor(S, X);
         if (found) {
            res.set(representative);
         }
      });
      return res;
   }
};
}
#endif //DB_DIALECTS_QUERYGRAPH_H
