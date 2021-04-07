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

namespace mlir::relalg {
class QueryGraph {
   public:
   class node_set {
      public:
      sul::dynamic_bitset<> storage;
      class bit_iterator
         : public std::iterator<std::forward_iterator_tag, size_t> {
         typedef bit_iterator iterator;
         sul::dynamic_bitset<>& bitset;
         size_t pos;

         public:
         bit_iterator(sul::dynamic_bitset<>& bitset, size_t pos) : bitset(bitset), pos(pos) {}
         ~bit_iterator() {}

         iterator operator++(int) /* postfix */ { return bit_iterator(bitset, bitset.find_next(pos)); }
         iterator& operator++() /* prefix */ {
            pos = bitset.find_next(pos);
            return *this;
         }
         reference operator*() { return pos; }
         pointer operator->() { return &pos; }
         bool operator==(const iterator& rhs) const { return pos == rhs.pos; }
         bool operator!=(const iterator& rhs) const { return pos != rhs.pos; }
      };

      public:
      node_set() {
      }
      node_set(size_t size) : storage(size) {}
      node_set negate() const {
         node_set res = *this;
         size_t pos = res.find_first();
         size_t flip_len = res.storage.size() - pos - 1;
         if (flip_len) {
            res.storage.flip(pos + 1, flip_len);
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
         res.storage.set(0, n + 1, true);
         return res;
      }

      static node_set single(size_t num_nodes, size_t pos) {
         auto res = node_set(num_nodes);
         res.set(pos);
         return res;
      }
      bool is_subset_of(const node_set& S) const{
         return storage.is_subset_of(S.storage);
      }
      bool intersects(const node_set& rhs) const{
         return storage.intersects(rhs.storage);
      }
      void set(size_t pos) {
         storage.set(pos);
      }
      auto begin() {
         return bit_iterator(storage, storage.find_first());
      }
      auto end() {
         return bit_iterator(storage, storage.npos);
      }
      size_t find_first() {
         return storage.find_first();
      }
      bool operator==(const node_set& rhs) const { return storage == rhs.storage; }
      bool operator!=(const node_set& rhs) const { return storage != rhs.storage; }
      bool operator<(const node_set& rhs) const { return storage < rhs.storage; }

      bool any() {
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
         const node_set& rhs) {
         node_set result = *this;
         result &= rhs;
         return result;
      }
      node_set operator~() const {
         node_set res = flip();
         return res;
      }
      node_set operator|(
         const node_set& rhs) {
         node_set result = *this;
         result |= rhs;
         return result;
      }
      bool valid() {
         return !storage.empty();
      }
      size_t count() {
         return storage.count();
      }
      node_set flip() const{
         node_set res = *this;
         res.storage.flip();
         return res;
      }
      void iterateSubsets(std::function<void(node_set)> fn) const {
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
      size_t hash() const {
         size_t res = 0;
         for (size_t i = 0; i < storage.num_blocks(); i++) {
            res ^= storage.data()[i];
         }
         return res;
      }
   };
   struct hash_node_set {
      size_t operator()(const node_set& bitset) const {
         return bitset.hash();
      }
   };
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;

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
   mlir::relalg::QueryGraph::node_set nodeSet(const std::vector<size_t>& nodes) {
      node_set res(num_nodes);
      for (auto x : nodes) {
         res.set(x);
      }
      return res;
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

   node_set expand(node_set S) {
      iterateNodes(S, [&](Node& n) {
         if (n.dependencies.valid()) {
            S |= n.dependencies;
         }
      });
      return S;
   }
   bool isConnected(llvm::EquivalenceClasses<size_t>& connections, node_set& S) {
      assert(S.any());
      size_t first_class = num_nodes;
      bool connected = true;
      for (auto pos : S) {
         if (first_class == num_nodes) {
            first_class = connections.getLeaderValue(pos);
         } else {
            if (first_class != connections.getLeaderValue(pos)) {
               connected = false;
            }
         }
      }
      return connected;
   }

   void ensureConnected() {
      llvm::EquivalenceClasses<size_t> already_connected;
      for (size_t i = 0; i < nodes.size(); i++) {
         already_connected.insert(i);
      }

      std::list<size_t> edges_to_process;
      for (size_t i = 0; i < edges.size(); i++) {
         if (edges[i].left.any() && edges[i].right.any()) {
            edges_to_process.push_back(i);
         }
      }
      for (size_t i = 0; i < edges.size(); i++) {
         std::list<size_t> new_list;
         for (auto edgeid : edges_to_process) {
            if (isConnected(already_connected, edges[edgeid].left) && isConnected(already_connected, edges[edgeid].right)) {
               already_connected.unionSets(edges[edgeid].left.find_first(), edges[edgeid].right.find_first());
            } else {
               new_list.push_back(edgeid);
            }
         }

         std::swap(edges_to_process, new_list);
      }
      for (auto& a : already_connected) {
         if (a.isLeader()) {
            for (auto& b : already_connected) {
               if (b.isLeader()) {
                  if (a.getData() != b.getData()) {
                     //std::cout << a.getData() << " vs " << b.getData() << "\n";
                     node_set left = getNodeSetFromClass(already_connected, a.getData());
                     if (expand(left) != left) {
                        continue;
                     }
                     node_set right = getNodeSetFromClass(already_connected, b.getData());
                     if (expand(right) != right) {
                        continue;
                     }
                     //std::cout << "would lead to edge (" << left << "," << right << ")\n";
                     bool connecting_edge_exists = false;
                     for (auto& edge : edges) {
                        if ((left.intersects(edge.left) && right.intersects(edge.right)) || (left.intersects(edge.right) && right.intersects(edge.left))) {
                           if (edge.op && !mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
                              //std::cout << "but prohibited by edge (" << edge.left << "," << edge.right << ")\n";
                              connecting_edge_exists = true;
                           }
                           break;
                        }
                        if (left == edge.left || right == edge.right || left == edge.right || right == edge.left) {
                           if (edge.op && !mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
                              //std::cout << "but prohibited by edge (" << edge.left << "," << edge.right << ")\n";
                              connecting_edge_exists = true;
                           }
                           break;
                        }
                     }
                     if (!connecting_edge_exists) {
                        addEdge(left, right, node_set(num_nodes), Operator(), EdgeType::REAL);
                     }
                  }
               }
            }
         }
      }
   }
   node_set getNodeSetFromClass(llvm::EquivalenceClasses<size_t> classes, size_t val) {
      node_set res(num_nodes);
      auto eqclass = classes.findLeader(val);
      for (auto me = classes.member_end(); eqclass != me; ++eqclass) {
         res.set(*eqclass);
      }
      return res;
   }
   node_set getNodeSetFromClasses(llvm::EquivalenceClasses<size_t> classes, node_set S) {
      node_set res(num_nodes);
      for (auto pos : S) {
         auto eqclass = classes.findLeader(pos);
         for (auto me = classes.member_end(); eqclass != me; ++eqclass) {
            res.set(*eqclass);
         }
      }
      return res;
   }
};
}
#endif //DB_DIALECTS_QUERYGRAPH_H
