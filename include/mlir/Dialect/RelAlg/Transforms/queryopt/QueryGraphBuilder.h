#ifndef MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPHBUILDER_H
#define MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPHBUILDER_H
#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

namespace mlir::relalg {
class QueryGraphBuilder {
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;

   Operator root;
   llvm::SmallPtrSet<mlir::Operation*,12>& alreadyOptimized;
   size_t numNodes;
   QueryGraph qg;
   class NodeResolver {
      QueryGraph& qg;

      std::unordered_map<const relalg::RelationalAttribute*, size_t> attrToNodes;

      public:
      explicit NodeResolver(QueryGraph& qg) : qg(qg) {}

      void add(const relalg::RelationalAttribute* attr, size_t nodeid) {
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

      size_t resolve(const relalg::RelationalAttribute* attr) {
         assert(attrToNodes.contains(attr));
         return attrToNodes[attr];
      }
   };
   size_t addPseudoNode(){
      return qg.addPseudoNode();
   }
   size_t addNode(Operator op) {
      QueryGraph::Node n(op);
      n.id = qg.nodes.size();
      qg.nodes.push_back(n);
      nodeForOp[op.getOperation()] = n.id;
      return n.id;
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

   void ensureConnected();


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
#endif // MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPHBUILDER_H
