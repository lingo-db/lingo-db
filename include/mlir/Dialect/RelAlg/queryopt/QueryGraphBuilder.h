#ifndef MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPHBUILDER_H
#define MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPHBUILDER_H
#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

namespace mlir::relalg {
class QueryGraphBuilder {
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;

   Operator root;
   llvm::SmallPtrSet<mlir::Operation*,12>& alreadyOptimized;
   size_t numNodes;
   QueryGraph qg;
   NodeSet emptyNode;

   class NodeResolver {
      QueryGraph& qg;

      std::unordered_map<relalg::RelationalAttribute*, size_t> attrToNodes;

      public:
      explicit NodeResolver(QueryGraph& qg) : qg(qg) {}

      void add(relalg::RelationalAttribute* attr, size_t nodeid) {
         attrToNodes[attr] = nodeid;
      }

      void merge(const NodeResolver& other) {
         for (auto x : other.attrToNodes) {
            auto [attr, nodeid] = x;
            if (attrToNodes.count(attr)) {
               auto currid = attrToNodes[attr];
               if (qg.nodes[nodeid].op->isBeforeInBlock(qg.nodes[currid].op.getOperation())) {
                  currid = nodeid;
               }
               attrToNodes[attr] = currid;
            } else {
               attrToNodes[attr] = nodeid;
            }
         }
      }

      size_t resolve(relalg::RelationalAttribute* attr) {
         return attrToNodes[attr];
      }
   };
   size_t addNode(Operator op) {
      QueryGraph::Node n(op);
      n.id = qg.nodes.size();
      qg.nodes.push_back(n);
      nodeForOp[op.getOperation()] = n.id;
      return n.id;
   }
   static std::pair<Operator, Operator> normalizeChildren(Operator op) {
      size_t left = 0;
      size_t right = 1;
      if (op->hasAttr("join_direction")) {
         mlir::relalg::JoinDirection joinDirection = mlir::relalg::symbolizeJoinDirection(
                                                        op->getAttr("join_direction").dyn_cast_or_null<mlir::IntegerAttr>().getInt())
                                                        .getValue();
         if (joinDirection == mlir::relalg::JoinDirection::right) {
            std::swap(left, right);
         }
      }
      return {op.getChildren()[left], op.getChildren()[right]};
   }
   static bool intersects(const attribute_set& a, const attribute_set& b) {
      return llvm::any_of(a, [&](auto x) { return b.contains(x); });
   }
   NodeSet calcTES(Operator op, NodeResolver& resolver);

   NodeResolver populateQueryGraph(Operator op);

   NodeSet calcSES(Operator op, NodeResolver& resolver) const;

   std::unordered_map<mlir::Operation*, NodeSet> ts;
   std::unordered_map<mlir::Operation*, NodeSet> teSs;
   std::unordered_map<mlir::Operation*, size_t> nodeForOp;

   NodeSet calcT(Operator op, NodeResolver& resolver) {
      if (ts.count(op.getOperation())) {
         return ts[op.getOperation()];
      } else {
         NodeSet init = NodeSet(numNodes);
         if (nodeForOp.count(op.getOperation())) {
            init.set(nodeForOp[op.getOperation()]);
         }
         if (!alreadyOptimized.count(op.getOperation())) {
            for (auto child : op.getChildren()) {
               init |= calcT(child, resolver);
            }
         }
         ts[op.getOperation()] = init;
         return init;
      }
   }
   bool canPushSelection(const NodeSet& ses, Operator curr, NodeResolver& resolver) {
      if (curr.getChildren().size() == 1) {
         return true;
      }
      auto tes = calcTES(curr, resolver);
      auto [b_left, b_right] = normalizeChildren(curr);
      NodeSet leftTes = calcT(b_left, resolver) & tes;
      NodeSet rightTes = calcT(b_right, resolver) & tes;
      if (leftTes.intersects(ses) && rightTes.intersects(ses)) {
         return false;
      }
      switch (mlir::relalg::detail::getBinaryOperatorType(curr)) {
         case detail::BinaryOperatorType::SemiJoin:
         case detail::BinaryOperatorType::MarkJoin:
         case detail::BinaryOperatorType::AntiSemiJoin:
         case detail::BinaryOperatorType::OuterJoin:
            return !rightTes.intersects(ses);
         case detail::BinaryOperatorType::FullOuterJoin: return false;
         default:
            return true;
      }
   }
   void ensureConnected();

   NodeSet expand(NodeSet s) {
      qg.iterateNodes(s, [&](QueryGraph::Node& n) {
         if (n.dependencies.valid()) {
            s |= n.dependencies;
         }
      });
      return s;
   }

   public:
   QueryGraphBuilder(Operator root, llvm::SmallPtrSet<mlir::Operation*,12>& alreadyOptimized);
   void generate() {
      populateQueryGraph(root);
      ensureConnected();
   }
   QueryGraph& getQueryGraph() {
      return qg;
   }
};
} // namespace mlir::relalg
#endif // MLIR_DIALECT_RELALG_QUERYOPT_QUERYGRAPHBUILDER_H
