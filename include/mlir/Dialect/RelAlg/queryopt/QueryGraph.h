#ifndef DB_DIALECTS_QUERYGRAPH_H
#define DB_DIALECTS_QUERYGRAPH_H

#include <functional>
#include <iostream>
#include "dynamic_bitset.h"


namespace mlir::relalg {
    class QueryGraph {
        size_t num_nodes;
    public:
        using node_set = sul::dynamic_bitset<>;
        struct Edge {
            node_set right;
            node_set left;
        };
        struct Node {
            size_t id;
            std::vector<size_t> edges;
        };
        std::vector<Node> nodes;
        std::vector<Edge> edges;

        QueryGraph(size_t num_nodes) : num_nodes(num_nodes) {
            for (size_t i = 0; i < num_nodes; i++) {
                Node n;
                n.id = i;
                nodes.push_back(n);
            }
        }

        void addEdge(std::vector<size_t> left, std::vector<size_t> right) {
            size_t edgeid = edges.size();
            edges.push_back(Edge());
            Edge &e = edges.back();
            e.left = node_set(num_nodes);
            e.right = node_set(num_nodes);

            for (auto n:left) {
                nodes[n].edges.push_back(edgeid);
                e.left.set(n);
            }
            for (auto n:right) {
                nodes[n].edges.push_back(edgeid);
                e.right.set(n);
            }
        }

        void iterateNodes(std::function<void(Node &)> fn) {
            iterateNodesDesc(fn);
        }

        void iterateNodesDesc(std::function<void(Node &)> fn) {
            for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
                fn(*it);
            }
        }

        void iterateSetDec(node_set S, std::function<void(size_t)> fn) {
            std::vector<size_t> positions;
            S.iterate_bits_on([&](size_t v) {
                positions.push_back(v);
            });
            for (auto it=positions.rbegin();it != positions.rend();it++) {
                fn(*it);
            }
        }


        bool isConnected(node_set S1, node_set S2) {
            bool found = false;
            S1.iterate_bits_on([&](size_t v) {
                Node &n = nodes[v];
                for (auto edgeid:n.edges) {
                    auto &edge = edges[edgeid];
                    if (edge.left.is_subset_of(S1) && edge.right.is_subset_of(S2)) {
                        found = true;
                    }
                    if (edge.left.is_subset_of(S2) && edge.right.is_subset_of(S1)) {
                        found = true;
                    }
                }
            });
            return found;
        }


        node_set getNeighbors(node_set S, node_set X) {
            node_set res(num_nodes);
            S.iterate_bits_on([&](size_t v) {
                Node &n = nodes[v];
                for (auto edgeid:n.edges) {
                    auto &edge = edges[edgeid];
                    if (edge.left.is_subset_of(S) && !S.intersects(edge.right) && !X.intersects(edge.right)) {
                        res.set(edge.right.find_first());
                    } else if (edge.right.is_subset_of(S) && !S.intersects(edge.left) && !X.intersects(edge.left)) {
                        res.set(edge.left.find_first());
                    }
                }
            });
            return res;
        }

        node_set fill_until(size_t n) {
            auto res = node_set(num_nodes);
            res.set(0, n + 1, true);
            return res;
        }

        node_set negate(node_set& S) {
            size_t pos = S.find_first();
            size_t flip_len = num_nodes - pos - 1;
            if (flip_len) {
                S.flip(pos + 1, flip_len);
            }
            return S;
        }

        node_set single(size_t pos) {
            auto res = node_set(num_nodes);
            res.set(pos);
            return res;
        }

        void iterateSubsets(node_set S, std::function<void(node_set)> fn) {
            if (!S.any())return;
            auto S1 = S & negate(S);
            do {
                fn(S1);
                auto S1flipped = S1;
                S1flipped.flip();
                auto S2 = S & S1flipped;
                S1 = S & negate(S2);
            } while (S1 != S);
        }
    };
}
#endif //DB_DIALECTS_QUERYGRAPH_H
