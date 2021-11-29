#include "mlir/Dialect/RelAlg/Transforms/queryopt/DPhyp.h"
#include <iostream>
#include <unordered_set>

void mlir::relalg::DPHyp::emitCsgCmp(const NodeSet& s1, const NodeSet& s2) {
   auto p1 = dpTable[s1];
   auto p2 = dpTable[s2];
   auto s = s1 | s2;
   struct HashOp {
      size_t operator()(const Operator& op) const {
         return (size_t) op.operator mlir::Operation*();
      }
   };
   std::unordered_set<Operator, HashOp> predicates;

   Operator specialJoin{};
   double totalSelectivity = 1;

   for (auto& edge : queryGraph.joins) {
      if (edge.connects(s1, s2)) {
         totalSelectivity *= edge.selectivity;
         if (!edge.op) {
            //special case: forced cross product
            //do nothing
         } else if (!mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
            specialJoin = edge.op;
         } else {
            predicates.insert(edge.op);
         }
         if (edge.createdNode) {
            s |= NodeSet::single(queryGraph.numNodes, edge.createdNode.getValue());
         }
      }
   }
   for (auto& edge : queryGraph.selections) {
      totalSelectivity*=queryGraph.calculateSelectivity(edge,s1,s2);
      if (edge.connects2(s, s1, s2)) {
         predicates.insert(edge.op);
      }
   }
   std::shared_ptr<Plan> currPlan;

   if (specialJoin) {
      double estimatedResultSize = p1->getRows() * p2->getRows() * totalSelectivity;
      if (mlir::isa<mlir::relalg::SemiJoinOp>(specialJoin.getOperation()) || mlir::isa<mlir::relalg::SemiJoinOp>(specialJoin.getOperation())) {
         estimatedResultSize = p1->getRows() * totalSelectivity;
      }
      if (mlir::isa<mlir::relalg::OuterJoinOp>(specialJoin.getOperation()) || mlir::isa<mlir::relalg::MarkJoinOp>(specialJoin.getOperation()) || mlir::isa<mlir::relalg::CollectionJoinOp>(specialJoin.getOperation()) || mlir::isa<mlir::relalg::SingleJoinOp>(specialJoin.getOperation())) {
         estimatedResultSize = p1->getRows();
      }

      currPlan = std::make_shared<Plan>(specialJoin, std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(predicates.begin(), predicates.end()), estimatedResultSize);
   } else if (!predicates.empty()) {
      auto estimatedResultSize = p1->getRows() * p2->getRows() * totalSelectivity;
      if (p1->getRows() > p2->getRows()) {
         std::swap(p1, p2);
      }
      currPlan = std::make_shared<Plan>(*predicates.begin(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(++predicates.begin(), predicates.end()), estimatedResultSize);
   } else {
      auto estimatedResultSize = p1->getRows() * p2->getRows() * totalSelectivity;
      currPlan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>({}), estimatedResultSize);
   }
   currPlan->setDescription("(" + p1->getDescription() + ") join (" + p2->getDescription() + ")");
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