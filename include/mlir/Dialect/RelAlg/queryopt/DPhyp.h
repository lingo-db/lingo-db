#ifndef DB_DIALECTS_DPHYP_H
#define DB_DIALECTS_DPHYP_H

#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <bitset>
#include <memory>

namespace mlir::relalg {
class DPHyp {
   std::unordered_map<node_set, std::shared_ptr<Plan>, hash_node_set> dp_table;
   QueryGraph& queryGraph;

   public:
   DPHyp(QueryGraph& qg) : queryGraph(qg) {}

   void emitCsg(node_set S1);

   void emitCsgCmp(const node_set& S1, const node_set& S2);

   void enumerateCsgRec(node_set S1, node_set X);

   void enumerateCmpRec(node_set S1, node_set S2, node_set X);

   std::shared_ptr<Plan> solve();
};
}

#endif //DB_DIALECTS_DPHYP_H
