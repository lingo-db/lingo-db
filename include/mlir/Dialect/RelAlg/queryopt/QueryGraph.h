#ifndef MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPH_H
#define MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPH_H

#include "llvm/Support/Debug.h"
#include <mlir/Dialect/DB/IR/DBOps.h>
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
      double selectivity = 1;
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
      double selectivity;
      double rows;

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
   Attributes getPKey(StringRef scope,mlir::Attribute pkeyAttribute){
      Attributes res;
      if(auto arrAttr=pkeyAttribute.dyn_cast_or_null<mlir::ArrayAttr>()){
         for(auto attr:arrAttr){
            if(auto stringAttr=attr.dyn_cast_or_null<mlir::StringAttr>()){
               auto relAttrManager=pkeyAttribute.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
               res.insert(relAttrManager.get(scope,stringAttr.getValue()).get());
            }
         }
      }
      return res;
   }

   void estimate() {
      for (auto& node : nodes) {
          node.selectivity=1;
          if(node.op) {
            node.rows=1;
            if(auto baseTableOp=mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())){
               if(baseTableOp->hasAttr("rows")){
                  node.rows=baseTableOp->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>().getInt();
               }
            }
            auto availableLeft = node.op.getAvailableAttributes();
            mlir::relalg::Attributes availableRight;
            std::vector<Predicate> predicates;
            for(auto pred:node.additionalPredicates) {
               addPredicates(predicates, pred, availableLeft, availableRight);
            }
            Attributes pkey;
            if(auto baseTableOp=mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())){
               pkey=getPKey(baseTableOp.sym_name(),baseTableOp->getAttr("pkey"));
            }
            Attributes predicatesLeft;
            for(auto predicate:predicates){
               predicatesLeft.insert(predicate.left);
            }
            bool pKeyIncluded =pkey.isSubsetOf(predicatesLeft);
            if(pKeyIncluded){
               node.selectivity=1/node.rows;
            }else {
               for (auto predicate : predicates) {
                  if(predicate.isEq){
                     node.selectivity*=0.1;
                  }else{
                     node.selectivity*=0.25;
                  }
               }
            }
         }
      }
      for (auto& edge : edges) {
         auto availableLeft = getAttributesForNodeSet(edge.left);
         auto availableRight = getAttributesForNodeSet(edge.right);
         std::vector<Predicate> predicates;
         addPredicates(predicates, edge.op, availableLeft, availableRight);
         edge.selectivity=1.0;
         std::vector<std::pair<double,Attributes>> pkeysLeft;
         std::vector<std::pair<double,Attributes>> pkeysRight;
         iterateNodes(edge.left,[&](auto node){
            if(node.op){
               if(auto baseTableOp=mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())){
                  pkeysLeft.push_back({node.rows,getPKey(baseTableOp.sym_name(),baseTableOp->getAttr("pkey"))});
               }
            }
         });
         iterateNodes(edge.right,[&](auto node){
           if(node.op){
              if(auto baseTableOp=mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())){
                 pkeysRight.push_back({node.rows,getPKey(baseTableOp.sym_name(),baseTableOp->getAttr("pkey"))});
              }
           }
         });

         Attributes predicatesLeft;
         Attributes predicatesRight;
         bool anyEq=false;
         for(auto predicate:predicates){
            predicatesLeft.insert(predicate.left);
            predicatesRight.insert(predicate.right);
            anyEq|=predicate.isEq;
         }
         for(auto p:pkeysLeft){
            auto [rows,pkey]=p;
            if(pkey.isSubsetOf(predicatesLeft)){
               edge.selectivity*=1/rows;
               predicatesLeft.remove(pkey);
            }
         }
         for(auto p:pkeysRight){
            auto [rows,pkey]=p;
            if(pkey.isSubsetOf(predicatesRight)){
               edge.selectivity*=1/rows;
               predicatesRight.remove(pkey);
            }
         }
         for(auto predicate:predicates){
            if(predicate.left.isSubsetOf(predicatesLeft)&&predicate.right.isSubsetOf(predicatesRight)){
               if(predicate.isEq){
                  edge.selectivity*=0.1;
               }else{
                  edge.selectivity*=0.25;
               }
            }
         }

      }
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPH_H
