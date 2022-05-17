#include "mlir/Dialect/RelAlg/Transforms/queryopt/DPhyp.h"
#include <unordered_set>

void mlir::relalg::DPHyp::emitCsgCmp(const NodeSet& s1, const NodeSet& s2) {
   auto p1 = dpTable[s1];
   auto p2 = dpTable[s2];
   NodeSet s;
   auto currPlan = Plan::joinPlans(s1, s2, p1, p2, queryGraph,s);
   if (!dpTable.count(s) || currPlan->getCost() < dpTable[s]->getCost()) {
      dpTable[s] = currPlan;
   }
}
void mlir::relalg::DPHyp::enumerateCsgRec(NodeSet s1, NodeSet x) {
   auto neighbors = queryGraph.getNeighbors(s1, x);
   neighbors.iterateSubsets([&](const NodeSet& n) {
      auto s1N = s1 | n;
      if (dpTable.count(s1N)) {
         emitCsg(s1N);
      }
   });
   neighbors.iterateSubsets([&](const NodeSet& n) {
      enumerateCsgRec(s1 | n, x | neighbors);
   });
}
void mlir::relalg::DPHyp::enumerateCmpRec(NodeSet s1, NodeSet s2, NodeSet x) {
   auto neighbors = queryGraph.getNeighbors(s2, x);
   neighbors.iterateSubsets([&](const NodeSet& n) {
      auto s2N = s2 | n;
      if (dpTable.count(s2N) && queryGraph.isConnected(s1, s2N)) {
         emitCsgCmp(s1, s2N);
      }
   });
   x = x | neighbors;
   neighbors.iterateSubsets([&](const NodeSet& n) {
      enumerateCmpRec(s1, s2 | n, x);
   });
}
void mlir::relalg::DPHyp::emitCsg(NodeSet s1) {
   NodeSet x = s1 | NodeSet::fillUntil(queryGraph.numNodes, s1.findFirst());
   auto neighbors = queryGraph.getNeighbors(s1, x);

   QueryGraph::iterateSetDec(neighbors, [&](size_t pos) {
      auto s2 = NodeSet::single(queryGraph.numNodes, pos);
      if (queryGraph.isConnected(s1, s2)) {
         emitCsgCmp(s1, s2);
      }
      enumerateCmpRec(s1, s2, x);
   });
}
static std::shared_ptr<mlir::relalg::Plan> createInitialPlan(mlir::relalg::QueryGraph::Node& n) {
   std::string description = std::to_string(n.id);
   if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(n.op.getOperation())) {
      description = baseTableOp.table_identifier().str();
   }
   auto currPlan = std::make_shared<mlir::relalg::Plan>(n.op, std::vector<std::shared_ptr<mlir::relalg::Plan>>({}), std::vector<Operator>({n.additionalPredicates}), n.rows * n.selectivity);
   currPlan->setDescription(description);
   return currPlan;
}

std::shared_ptr<mlir::relalg::Plan> mlir::relalg::DPHyp::solve() {
   for (auto v : queryGraph.getNodes()) {
      dpTable.insert({NodeSet::single(queryGraph.numNodes, v.id), createInitialPlan(v)});
   }
   queryGraph.iterateNodesDesc([&](QueryGraph::Node& v) {
      auto onlyV = NodeSet::single(queryGraph.numNodes, v.id);
      emitCsg(onlyV);
      auto bv = NodeSet::fillUntil(queryGraph.numNodes, v.id);
      enumerateCsgRec(onlyV, bv);
   });
   return dpTable[NodeSet::ones(queryGraph.numNodes)];
}
void mlir::relalg::DPHyp::countSubGraphsRec(NodeSet s1, NodeSet x, size_t& count, size_t maxCount) {
   auto neighbors = queryGraph.getNeighbors(s1, x);
   neighbors.iterateSubsets([&](const NodeSet& n) {
      if ((++count) >= maxCount) return;
      countSubGraphsRec(s1 | n, x | neighbors, count, maxCount);
   });
}

size_t mlir::relalg::DPHyp::countSubGraphs(size_t maxCount) {
   size_t count = 0;
   queryGraph.iterateNodesDesc([&](QueryGraph::Node& v) {
      auto onlyV = NodeSet::single(queryGraph.numNodes, v.id);
      if ((++count) >= maxCount) return;
      auto bv = NodeSet::fillUntil(queryGraph.numNodes, v.id);
      countSubGraphsRec(onlyV, bv, count, maxCount);
   });
   return count;
}