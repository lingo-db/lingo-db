#ifndef MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPHBUILDER_H
#define MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_QUERYGRAPHBUILDER_H
#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

namespace mlir::relalg {
class QueryGraphBuilder {

   Operator root;
   llvm::SmallPtrSet<mlir::Operation*,12>& alreadyOptimized;
   size_t numNodes;
   QueryGraph qg;
   std::unordered_map<const relalg::Column*, size_t> attrToNodes;

   size_t addNode(Operator op) {
      QueryGraph::Node n(op);
      n.id = qg.nodes.size();
      qg.nodes.push_back(n);
      nodeForOp[op.getOperation()] = n.id;
      return n.id;
   }

   NodeSet calcTES(Operator op);

   void populateQueryGraph(Operator op);

   NodeSet calcSES(Operator op) const;

   std::unordered_map<mlir::Operation*, NodeSet> ts;
   std::unordered_map<mlir::Operation*, NodeSet> teSs;
   std::unordered_map<mlir::Operation*, size_t> nodeForOp;

   NodeSet calcT(Operator op) {
      if (ts.count(op.getOperation())) {
         return ts[op.getOperation()];
      } else {
         NodeSet init = NodeSet(numNodes);
         if (nodeForOp.count(op.getOperation())) {
            init.set(nodeForOp[op.getOperation()]);
         }
         if (!alreadyOptimized.count(op.getOperation())) {
            for (auto child : op.getChildren()) {
               init |= calcT(child);
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
