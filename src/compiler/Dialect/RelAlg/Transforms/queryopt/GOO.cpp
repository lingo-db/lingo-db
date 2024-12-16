#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/GOO.h"
using namespace lingodb::compiler::dialect;
std::shared_ptr<relalg::Plan> relalg::GOO::createInitialPlan(relalg::QueryGraph::Node& n) {
   std::string description = std::to_string(n.id);
   if (auto baseTableOp = mlir::dyn_cast_or_null<relalg::BaseTableOp>(n.op.getOperation())) {
      description = baseTableOp.getTableIdentifier().str();
   }
   auto currPlan = std::make_shared<relalg::Plan>(n.op, std::vector<std::shared_ptr<relalg::Plan>>({}), std::vector<Operator>({n.additionalPredicates}), n.rows * n.selectivity);
   currPlan->setDescription(description);
   return currPlan;
}
std::shared_ptr<relalg::Plan> relalg::GOO::solve() {
   std::vector<std::pair<NodeSet, std::shared_ptr<relalg::Plan>>> availablePlans;
   for (auto v : queryGraph.getNodes()) {
      availablePlans.push_back({NodeSet::single(queryGraph.numNodes, v.id), createInitialPlan(v)});
   }
   while (availablePlans.size() > 1) {
      size_t leftIdx;
      size_t rightIdx;
      double best = std::numeric_limits<double>::max();
      NodeSet newProblem;
      std::shared_ptr<Plan> bestPlan;
      for (size_t l = 0; l < availablePlans.size(); l++) {
         for (size_t r = 0; r < l; r++) {
            auto [leftProblem, leftPlan] = availablePlans[l];
            auto [rightProblem, rightPlan] = availablePlans[r];
            if (queryGraph.isConnected(leftProblem, rightProblem)) {
               NodeSet s;
               auto currPlan = Plan::joinPlans(leftProblem, rightProblem, leftPlan, rightPlan, queryGraph, s);
               if (currPlan->getRows() < best) {
                  bestPlan = currPlan;
                  best = currPlan->getRows();
                  leftIdx = l;
                  rightIdx = r;
                  newProblem = s;
               }
            }
         }
      }
      if (!bestPlan) {
         return {};
      }
      //remove previous plans
      if (leftIdx > rightIdx) {
         std::swap(leftIdx, rightIdx);
      }
      availablePlans.erase(availablePlans.begin() + rightIdx);
      availablePlans.erase(availablePlans.begin() + leftIdx);

      //add best plan instead;
      availablePlans.push_back({newProblem, bestPlan});
   }
   return availablePlans[0].second;
}