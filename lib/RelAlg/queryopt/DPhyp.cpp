#include "mlir/Dialect/RelAlg/queryopt/DPhyp.h"

void mlir::relalg::DPHyp::emitCsgCmp(const node_set& s1, const node_set& s2) {
   auto p1 = dp_table[s1];
   auto p2 = dp_table[s2];
   auto s = s1 | s2;
   struct HashOp {
      size_t operator()(const Operator& op) const {
         return (size_t) op.operator mlir::Operation*();
      }
   };
   std::unordered_set<Operator, HashOp> predicates;
   std::unordered_set<Operator, HashOp> singlePredicates;

   Operator implicitOperator{};
   Operator specialJoin{};
   bool ignore = false;
   bool edgeInverted = false;
   for (auto& edge : queryGraph.edges) {
      if (edge.connects(s1, s2)) {
         edgeInverted = (edge.left.is_subset_of(s2) && edge.right.is_subset_of(s1)); //todo also include arbitrary?
         if (edge.edgeType == QueryGraph::EdgeType::IMPLICIT) {
            auto& implicitNode = queryGraph.nodes[edge.right.find_first()];
            implicitOperator = implicitNode.op;
            predicates.insert(implicitNode.additional_predicates.begin(), implicitNode.additional_predicates.end());
         } else if (edge.edgeType == QueryGraph::EdgeType::IGNORE) {
            ignore = true;
         } else if (!edge.op) {
            //special case: forced cross product
            //do nothing
         } else if (!mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
            specialJoin = edge.op;
         } else {
            predicates.insert(edge.op);
         }
      } else if ((edge.left | edge.right | edge.arbitrary).is_subset_of(s1 | s2) && !(edge.left | edge.right | edge.arbitrary).is_subset_of(s1) && !(edge.left | edge.right | edge.arbitrary).is_subset_of(s2)) {
         if (edge.op && (mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) || mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation()))) {
            singlePredicates.insert(edge.op);
         }
      }
   }
   std::shared_ptr<Plan> currPlan;
   predicates.insert(singlePredicates.begin(), singlePredicates.end());
   if (ignore) {
      auto child = edgeInverted ? p2 : p1;
      currPlan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({child}), std::vector<Operator>(predicates.begin(), predicates.end()), 0);
   } else if (implicitOperator) {
      auto subplans = std::vector<std::shared_ptr<Plan>>({p1});
      if (edgeInverted) {
         subplans = std::vector<std::shared_ptr<Plan>>({p2});
      }
      currPlan = std::make_shared<Plan>(implicitOperator, subplans, std::vector<Operator>(predicates.begin(), predicates.end()), 0);
   } else if (specialJoin) {
      currPlan = std::make_shared<Plan>(specialJoin, std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(predicates.begin(), predicates.end()), 0);
   } else if (!predicates.empty()) {
      currPlan = std::make_shared<Plan>(*predicates.begin(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(++predicates.begin(), predicates.end()), 0);
   } else {
      currPlan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>({}), 0);
   }
   currPlan->setDescription("(" + p1->getDescription() + ") join (" + p2->getDescription() + ")");

   if (!dp_table.count(s) || currPlan->getCost() < dp_table[s]->getCost()) {
      dp_table[s] = currPlan;
   }
}
void mlir::relalg::DPHyp::enumerateCsgRec(node_set s1, node_set x) {
   auto neighbors = queryGraph.getNeighbors(s1, x);
   neighbors.iterateSubsets([&](const node_set& n) {
      auto s1N = s1 | n;
      if (dp_table.count(s1N)) {
         emitCsg(s1N);
      }
   });
   neighbors.iterateSubsets([&](const node_set& n) {
      enumerateCsgRec(s1 | n, x | neighbors);
   });
}
void mlir::relalg::DPHyp::enumerateCmpRec(node_set s1, node_set s2, node_set x) {
   auto neighbors = queryGraph.getNeighbors(s2, x);
   neighbors.iterateSubsets([&](const node_set& n) {
      auto s2N = s2 | n;
      if (dp_table.count(s2N) && queryGraph.isConnected(s1, s2N)) {
         emitCsgCmp(s1, s2N);
      }
   });
   x = x | neighbors;
   neighbors.iterateSubsets([&](const node_set& n) {
      enumerateCmpRec(s1, s2 | n, x);
   });
}
void mlir::relalg::DPHyp::emitCsg(node_set s1) {
   node_set x = s1 | node_set::fill_until(queryGraph.num_nodes, s1.find_first());
   auto neighbors = queryGraph.getNeighbors(s1, x);

   QueryGraph::iterateSetDec(neighbors, [&](size_t pos) {
      auto s2 = node_set::single(queryGraph.num_nodes, pos);
      if (queryGraph.isConnected(s1, s2)) {
         emitCsgCmp(s1, s2);
      }
      enumerateCmpRec(s1, s2, x);
   });
}
static std::shared_ptr<mlir::relalg::Plan> createInitialPlan(mlir::relalg::QueryGraph::Node& n) {
   auto currPlan = std::make_shared<mlir::relalg::Plan>(n.op, std::vector<std::shared_ptr<mlir::relalg::Plan>>({}), std::vector<Operator>({n.additional_predicates}), 0);
   currPlan->setDescription(std::to_string(n.id));
   return currPlan;
}

std::shared_ptr<mlir::relalg::Plan> mlir::relalg::DPHyp::solve() {
   for (auto v : queryGraph.getNodes()) {
      dp_table.insert({node_set::single(queryGraph.num_nodes, v.id), createInitialPlan(v)});
   }
   queryGraph.iterateNodesDesc([&](QueryGraph::Node& v) {
      auto onlyV = node_set::single(queryGraph.num_nodes, v.id);
      emitCsg(onlyV);
      auto bv = node_set::fill_until(queryGraph.num_nodes, v.id);
      enumerateCsgRec(onlyV, bv);
   });
   return dp_table[node_set::ones(queryGraph.num_nodes)];
}