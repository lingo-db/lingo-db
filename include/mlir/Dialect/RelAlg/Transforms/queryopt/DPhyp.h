#ifndef MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_DPHYP_H
#define MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_DPHYP_H

#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <bitset>
#include <memory>

namespace mlir::relalg {
class DPHyp {
   std::unordered_map<NodeSet, std::shared_ptr<Plan>, HashNodeSet> dpTable;
   QueryGraph& queryGraph;

   public:
   DPHyp(QueryGraph& qg) : queryGraph(qg) {}

   void emitCsg(NodeSet s1);

   void emitCsgCmp(const NodeSet& s1, const NodeSet& s2);

   void enumerateCsgRec(NodeSet s1, NodeSet x);

   void enumerateCmpRec(NodeSet s1, NodeSet s2, NodeSet x);

   std::shared_ptr<Plan> solve();
   void countSubGraphsRec(NodeSet s1, NodeSet x, size_t& count, size_t maxCount);
   size_t countSubGraphs(size_t maxCount);
};
} // namespace mlir::relalg

#endif // MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_DPHYP_H
