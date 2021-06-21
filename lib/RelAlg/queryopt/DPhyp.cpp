#include "mlir/Dialect/RelAlg/queryopt/DPhyp.h"
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
   std::unordered_set<Operator, HashOp> singlePredicates;

   Operator implicitOperator{};
   Operator specialJoin{};
   bool ignore = false;
   bool edgeInverted = false;
   double totalSelectivity=1;
   bool hashImpl=false;
   for (auto& edge : queryGraph.edges) {
      if (edge.connects(s1, s2)) {
         hashImpl|=edge.hashImpl;
         totalSelectivity*=edge.selectivity;
         edgeInverted = (edge.left.isSubsetOf(s2) && edge.right.isSubsetOf(s1)); //todo also include arbitrary?
         if (edge.edgeType == QueryGraph::EdgeType::IMPLICIT) {
            auto& implicitNode = queryGraph.nodes[edge.right.findFirst()];
            implicitOperator = implicitNode.op;
            predicates.insert(implicitNode.additionalPredicates.begin(), implicitNode.additionalPredicates.end());
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
      } else if ((edge.left | edge.right | edge.arbitrary).isSubsetOf(s1 | s2) && !(edge.left | edge.right | edge.arbitrary).isSubsetOf(s1) && !(edge.left | edge.right | edge.arbitrary).isSubsetOf(s2)) {
         if (edge.op && (mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) || mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation()))) {
            singlePredicates.insert(edge.op);
            totalSelectivity*=edge.selectivity;
            hashImpl|=edge.hashImpl;
         }
      }
   }
   llvm::dbgs()<<"totalSelectivity:"<<totalSelectivity<<"\n";
   std::shared_ptr<Plan> currPlan;
   predicates.insert(singlePredicates.begin(), singlePredicates.end());

   if (ignore) {
      auto child = edgeInverted ? p2 : p1;
      currPlan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({child}), std::vector<Operator>(predicates.begin(), predicates.end()), child->getRows());
   } else if (implicitOperator) {
      auto subplans = std::vector<std::shared_ptr<Plan>>({p1});
      if (edgeInverted) {
         subplans = std::vector<std::shared_ptr<Plan>>({p2});
      }
      currPlan = std::make_shared<Plan>(implicitOperator, subplans, std::vector<Operator>(predicates.begin(), predicates.end()), subplans[0]->getRows());
   } else if (specialJoin) {
      auto estimatedResultSize=p1->getRows()*p2->getRows()*totalSelectivity;
      currPlan = std::make_shared<Plan>(specialJoin, std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(predicates.begin(), predicates.end()), estimatedResultSize,hashImpl);
   } else if (!predicates.empty()) {
      auto estimatedResultSize=p1->getRows()*p2->getRows()*totalSelectivity;
      currPlan = std::make_shared<Plan>(*predicates.begin(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(++predicates.begin(), predicates.end()), estimatedResultSize,hashImpl);
   } else {
      auto estimatedResultSize=p1->getRows()*p2->getRows()*totalSelectivity;
      currPlan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>({}), estimatedResultSize,hashImpl);
   }
   currPlan->setDescription("(" + p1->getDescription() + ") join (" + p2->getDescription() + ")");
   llvm::dbgs()<<currPlan->getDescription()<<" costs "<<currPlan->getCost()<<" size "<<currPlan->getRows()<<"\n";
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
   std::string description=std::to_string(n.id);
   if(auto baseTableOp=mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(n.op.getOperation())){
      description=baseTableOp.table_identifier().str();
   }
   auto currPlan = std::make_shared<mlir::relalg::Plan>(n.op, std::vector<std::shared_ptr<mlir::relalg::Plan>>({}), std::vector<Operator>({n.additionalPredicates}), n.rows*n.selectivity);
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
   llvm::dbgs()<<"final:"<<dpTable[NodeSet::ones(queryGraph.numNodes)]->getDescription()<<" costs: "<<dpTable[NodeSet::ones(queryGraph.numNodes)]->getCost()<<" size: "<<dpTable[NodeSet::ones(queryGraph.numNodes)]->getRows()<<"\n";
   return dpTable[NodeSet::ones(queryGraph.numNodes)];
}