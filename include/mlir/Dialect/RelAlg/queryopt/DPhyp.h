#ifndef DB_DIALECTS_DPHYP_H
#define DB_DIALECTS_DPHYP_H

#include "QueryGraph.h"
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <bitset>
#include <memory>

namespace mlir::relalg {
class PlanVisualizer {
   //std::string addNode();
};

struct Plan {
   Plan(Operator op, std::vector<std::shared_ptr<Plan>> subplans, std::vector<Operator> additional_ops, size_t cost) : op(op), subplans(subplans), additional_ops(additional_ops), cost(cost) {}
   Operator op;
   std::vector<Operator> additional_ops;
   std::vector<std::shared_ptr<Plan>> subplans;
   size_t cost;
   std::string descr;
};

class CostFunction {
};

class DPHyp {
   using node_set = QueryGraph::node_set;
   std::unordered_map<node_set, std::shared_ptr<Plan>, QueryGraph::hash_dyn_bitset> dp_table;

   QueryGraph& queryGraph;
   CostFunction& costFunction;

   std::shared_ptr<Plan> createInitialPlan(QueryGraph::Node& n) {
      auto curr_plan = std::make_shared<Plan>(n.op, std::vector<std::shared_ptr<Plan>>({}), std::vector<Operator>({n.additional_predicates}), 0);
      curr_plan->descr = std::to_string(n.id);
      return curr_plan;
   }

   public:
   DPHyp(QueryGraph& qg, CostFunction& costFunction) : queryGraph(qg), costFunction(costFunction) {}

   void EmitCsg(node_set S1) {
      node_set X = S1 | queryGraph.fill_until(S1.find_first());
      auto neighbors = queryGraph.getNeighbors(S1, X);

      queryGraph.iterateSetDec(neighbors, [&](size_t pos) {
         auto S2 = queryGraph.single(pos);
         if (queryGraph.isConnected(S1, S2)) {
            EmitCsgCmp(S1, S2);
         }
         EnumerateCmpRec(S1, S2, X);
      });
   }

   void EmitCsgCmp(node_set S1, node_set S2) {
      auto p1 = dp_table[S1];
      auto p2 = dp_table[S2];
      auto S = S1 | S2;
      struct hash_op {
         size_t operator()(const Operator& op) const {
            return (size_t) op.operator mlir::Operation*();
         }
      };
      std::unordered_set<Operator, hash_op> predicates;
      std::unordered_set<Operator, hash_op> single_predicates;

      Operator implicit_operator{};
      Operator special_join{};
      bool ignore = false;
      bool edgeInverted = false;
      for (auto& edge : queryGraph.edges) {
         if (edge.connects(S1, S2)) {
            edgeInverted = (edge.left.is_subset_of(S2) && edge.right.is_subset_of(S1)); //todo
            if (edge.edgeType == QueryGraph::EdgeType::IMPLICIT) {
               auto& implicit_node = queryGraph.nodes[edge.right.find_first()];
               implicit_operator = implicit_node.op;
               predicates.insert(implicit_node.additional_predicates.begin(), implicit_node.additional_predicates.end());
            } else if (edge.edgeType == QueryGraph::EdgeType::IGNORE) {
               ignore = true;
            } else if (!edge.op) {
               //special case: forced cross product
               //do nothing
            } else if (!mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
               special_join = edge.op;
               predicates.insert(edge.additional_predicates.begin(), edge.additional_predicates.end());
            } else {
               predicates.insert(edge.op);
               predicates.insert(edge.additional_predicates.begin(), edge.additional_predicates.end());
            }
         } else if ((edge.left | edge.right | edge.arbitrary).is_subset_of(S1 | S2)) {
            if (edge.op && mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation())) {
               single_predicates.insert(edge.op);
            }
         }
      }
      std::shared_ptr<Plan> curr_plan;
      if (ignore) {
         predicates.insert(single_predicates.begin(), single_predicates.end());
         auto child = edgeInverted ? p2 : p1;
         curr_plan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({child}), std::vector<Operator>(predicates.begin(), predicates.end()), 0);
      } else if (implicit_operator) {
         //predicates.insert(predicates.end(),single_predicates.begin(),single_predicates.end());
         auto subplans = std::vector<std::shared_ptr<Plan>>({p1});
         if (edgeInverted) {
            subplans = std::vector<std::shared_ptr<Plan>>({p2});
         }
         curr_plan = std::make_shared<Plan>(implicit_operator, subplans, std::vector<Operator>(predicates.begin(), predicates.end()), 0);
      } else if (special_join) {
         curr_plan = std::make_shared<Plan>(special_join, std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(predicates.begin(), predicates.end()), 0);
      } else if (!predicates.empty()) {
         curr_plan = std::make_shared<Plan>(*predicates.begin(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>(++predicates.begin(), predicates.end()), 0);
      } else {
         curr_plan = std::make_shared<Plan>(Operator(), std::vector<std::shared_ptr<Plan>>({p1, p2}), std::vector<Operator>({}), 0);
      }
      curr_plan->descr = "(" + p1->descr + ") join (" + p2->descr + ")";

      if (!dp_table.count(S) || curr_plan->cost < dp_table[S]->cost) {
         dp_table[S] = curr_plan;
      }
   }

   void EnumerateCsgRec(node_set S1, node_set X) {
      auto neighbors = queryGraph.getNeighbors(S1, X);
      queryGraph.iterateSubsets(neighbors, [&](node_set N) {
         auto S1N = S1 | N;
         if (dp_table.count(S1N)) {
            EmitCsg(S1N);
         }
      });
      queryGraph.iterateSubsets(neighbors, [&](node_set N) {
         EnumerateCsgRec(S1 | N, X | neighbors);
      });
   }

   void EnumerateCmpRec(node_set S1, node_set S2, node_set X) {
      auto neighbors = queryGraph.getNeighbors(S2, X);
      queryGraph.iterateSubsets(neighbors, [&](node_set N) {
         auto S2N = S2 | N;
         if (dp_table.count(S2N) && queryGraph.isConnected(S1, S2N)) {
            EmitCsgCmp(S1, S2N);
         }
      });
      X = X | neighbors;
      queryGraph.iterateSubsets(neighbors, [&](node_set N) {
         EnumerateCmpRec(S1, S2 | N, X);
      });
   }

   std::shared_ptr<Plan> solve() {
      queryGraph.iterateNodes([&](QueryGraph::Node& v) {
         dp_table.insert({queryGraph.single(v.id), createInitialPlan(v)});
      });
      queryGraph.iterateNodesDesc([&](QueryGraph::Node& v) {
         auto only_v = queryGraph.single(v.id);
         EmitCsg(only_v);
         auto Bv = queryGraph.fill_until(v.id);
         EnumerateCsgRec(only_v, Bv);
      });
      return dp_table[queryGraph.ones()];
   }
};
}

#endif //DB_DIALECTS_DPHYP_H
