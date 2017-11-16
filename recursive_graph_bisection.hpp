#pragma once

#include <cstdint>

namespace constants {
const uint64_t MAX_DEPTH = 13;
const uint64_t MAX_ITERATIONS = 10;
}

struct docid_node {
    uint64_t initial_id;
};

struct query_node {
    std::vector<docid_node*> postings;
};

struct bipartite_graph {
    std::vector<docid_node> G;
    std::vector<query_node> Q;
};

void recursive_bisection(std::vector<query_node>& Q, docid_node* G, size_t n, uint64_t depth = 0)
{
    // (1) create the initial partition
    auto V1, V2 = initial_partition(G, n);

    // (2) perform bisection
    for (uint64_t cur_iter = 1; cur_iter <= constants::MAX_ITERATIONS; cur_iter++) {
        // (2a) compute move gains. this has to happen in O(m) time, so O(1) per query
        auto gains_v1, gains_v2 = compute_move_gains(Q, V1, V2);

        // (2a) sort by decreasing gain. O(n log n)
        std::sort(gains_v1.begin(), gains_v1.end());
        std::sort(gains_v2.begin(), gains_v2.end());

        // (2b) swap. O(n)
        size_t num_swaps = 0;
        auto itr_v1 = gains_v1.begin();
        auto itr_v2 = gains_v2.begin();
        while (itr_v1 != gains_v1.end() && itr_v2 != gains_v2.end()) {
            if (itr_v1.gain + itr_v2.gain > 0) {
                // maybe we need to do something here to make compute_move_gains() efficient?
                swap_nodes(itr_v1.node, itr_v2.node);
                num_swaps++;
            } else {
                break;
            }
        }

        // (2c) converged?
        if (num_swaps == 0) {
            break;
        }
    }

    // (4) recurse. at most O(log n) recursion steps
    if (depth + 1 <= constants::MAX_DEPTH) {
        if (V1.size > 1)
            recursive_bisection(Q, V1, V1.size, depth + 1);
        if (V2.size > 1)
            recursive_bisection(Q, V2, V1.size, depth + 1);
    }
}

docid_mapping reorder_docids_graph_bisection(inverted_index& invidx)
{
    auto bipartite_graph = construct_bipartite_graph(invidx);

    recursive_bisection(bipartite_graph.Q, bipartite_graph.G);

    return create_mapping(Q, G);
}
