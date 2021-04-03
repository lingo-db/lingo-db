#ifndef DB_DIALECTS_DPHYP_H
#define DB_DIALECTS_DPHYP_H

#include <bitset>
#include <memory>
#include "QueryGraph.h"

namespace mlir::relalg {
    class PlanVisualizer {
        //std::string addNode();
    };

    struct Plan {
        Plan(std::string p) : plan(p) {}

        std::string plan;
        size_t cost;
    };

    class CostFunction {

    };

    class DPHyp {


        using node_set = QueryGraph::node_set;
        std::unordered_map<node_set, std::shared_ptr<Plan>, QueryGraph::hash_dyn_bitset> dp_table;

        QueryGraph &queryGraph;
        CostFunction &costFunction;

        std::shared_ptr<Plan> createInitialPlan(QueryGraph::Node &n) {
            return std::make_shared<Plan>(std::to_string(n.id));
        }


    public:
        DPHyp(QueryGraph &qg, CostFunction &costFunction) : queryGraph(qg), costFunction(costFunction) {}

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
            auto newplan = std::make_shared<Plan>("("+p1->plan+") join ("+p2->plan+")");//todo
            std::cout<<"newplan("<<S<<")="<<newplan->plan<<std::endl;
            if (!dp_table.count(S) || newplan->cost < dp_table[S]->cost) {
                dp_table[S] = newplan;
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

        void solve() {
            queryGraph.iterateNodes([&](QueryGraph::Node &v) {
                dp_table.insert({queryGraph.single(v.id), createInitialPlan(v)});
            });
            queryGraph.iterateNodesDesc([&](QueryGraph::Node &v) {
                auto only_v = queryGraph.single(v.id);
                EmitCsg(only_v);
                auto Bv = queryGraph.fill_until(v.id);
                EnumerateCsgRec(only_v, Bv);
            });

        }
    };
}

#endif //DB_DIALECTS_DPHYP_H
