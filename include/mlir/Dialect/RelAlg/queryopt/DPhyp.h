#ifndef DB_DIALECTS_DPHYP_H
#define DB_DIALECTS_DPHYP_H

#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <bitset>
#include <memory>

namespace mlir::relalg {
struct Plan {
   Plan(Operator op, const std::vector<std::shared_ptr<Plan>>& subplans, const std::vector<Operator>& additional_ops, size_t cost) : op(op), subplans(subplans), additional_ops(additional_ops), cost(cost) {}
   Operator op;
   std::vector<std::shared_ptr<Plan>> subplans;
   std::vector<Operator> additional_ops;
   size_t cost;
   std::string descr;
};

class DPHyp {
   std::unordered_map<node_set, std::shared_ptr<Plan>, hash_node_set> dp_table;
   QueryGraph& queryGraph;

   public:
   DPHyp(QueryGraph& qg) : queryGraph(qg) {}

   void EmitCsg(node_set S1);

   void EmitCsgCmp(const node_set& S1, const node_set& S2);

   void EnumerateCsgRec(node_set S1, node_set X);

   void EnumerateCmpRec(node_set S1, node_set S2, node_set X);

   std::shared_ptr<Plan> solve();
};
}

#endif //DB_DIALECTS_DPHYP_H
