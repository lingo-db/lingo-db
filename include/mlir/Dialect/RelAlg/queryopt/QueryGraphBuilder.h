#ifndef DB_DIALECTS_QUERYGRAPHBUILDER_H
#define DB_DIALECTS_QUERYGRAPHBUILDER_H
#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

namespace mlir::relalg {
class QueryGraphBuilder {
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;

   Operator root;
   std::unordered_set<mlir::Operation*>& already_optimized;
   size_t num_nodes;
   QueryGraph qg;
   node_set empty_node;

   class NodeResolver {
      QueryGraph& qg;

      std::unordered_map<relalg::RelationalAttribute*, size_t> attr_to_nodes;

      public:
      explicit NodeResolver(QueryGraph& qg) : qg(qg) {}

      void add(relalg::RelationalAttribute* attr, size_t nodeid) {
         attr_to_nodes[attr] = nodeid;
      }

      void merge(const NodeResolver& other) {
         for (auto x : other.attr_to_nodes) {
            auto [attr, nodeid] = x;
            if (attr_to_nodes.count(attr)) {
               auto currid = attr_to_nodes[attr];
               if (qg.nodes[nodeid].op->isBeforeInBlock(qg.nodes[currid].op.getOperation())) {
                  currid = nodeid;
               }
               attr_to_nodes[attr] = currid;
            } else {
               attr_to_nodes[attr] = nodeid;
            }
         }
      }

      size_t resolve(relalg::RelationalAttribute* attr) {
         return attr_to_nodes[attr];
      }
   };
   size_t addNode(Operator op) {
      QueryGraph::Node n(op);
      n.id = qg.nodes.size();
      qg.nodes.push_back(n);
      node_for_op[op.getOperation()] = n.id;
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
   node_set calcTES(Operator b, NodeResolver& resolver);

   NodeResolver populateQueryGraph(Operator op);

   node_set calcSES(Operator op, NodeResolver& resolver) const;

   std::unordered_map<mlir::Operation*, node_set> Ts;
   std::unordered_map<mlir::Operation*, node_set> TESs;
   std::unordered_map<mlir::Operation*, size_t> node_for_op;

   node_set calcT(Operator op, NodeResolver& resolver) {
      if (Ts.count(op.getOperation())) {
         return Ts[op.getOperation()];
      } else {
         node_set init = node_set(num_nodes);
         if (node_for_op.count(op.getOperation())) {
            init.set(node_for_op[op.getOperation()]);
         }
         if (!already_optimized.count(op.getOperation())) {
            for (auto child : op.getChildren()) {
               init |= calcT(child, resolver);
            }
         }
         Ts[op.getOperation()] = init;
         return init;
      }
   }
   bool canPushSelection(const node_set& SES, Operator curr, NodeResolver& resolver) {
      if (curr.getChildren().size() == 1) {
         return true;
      }
      auto TES = calcTES(curr, resolver);
      auto [b_left, b_right] = normalizeChildren(curr);
      node_set left_TES = calcT(b_left, resolver) & TES;
      node_set right_TES = calcT(b_right, resolver) & TES;
      if (left_TES.intersects(SES) && right_TES.intersects(SES)) {
         return false;
      }
      switch (mlir::relalg::detail::getBinaryOperatorType(curr)) {
         case detail::SemiJoin:
         case detail::MarkJoin:
         case detail::AntiSemiJoin:
         case detail::OuterJoin:
            return !right_TES.intersects(SES);
         case detail::FullOuterJoin: return false;
         default:
            return true;
      }
   }
   void ensureConnected();

   node_set expand(node_set S) {
      qg.iterateNodes(S, [&](QueryGraph::Node& n) {
         if (n.dependencies.valid()) {
            S |= n.dependencies;
         }
      });
      return S;
   }

   public:
   QueryGraphBuilder(Operator root, std::unordered_set<mlir::Operation*>& already_optimized);
   void generate() {
      populateQueryGraph(root);
      ensureConnected();
   }
   QueryGraph& getQueryGraph() {
      return qg;
   }
};
}
#endif //DB_DIALECTS_QUERYGRAPHBUILDER_H
