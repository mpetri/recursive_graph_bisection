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

struct partition_t {
    docid_node* V1;
    docid_node* V2;
    size_t n1;
    size_t n2;
};

/* random shuffle seems to do ok */
partition_t initial_partition(docid_node* G, size_t n)
{
    partition_t p;
    std::mt19937 rnd(n);
    std::shuffle(G, G + n, rnd);
    p.V1 = G;
    p.n1 = (n / 2);
    p.V2 = G + p.n1;
    p.n2 = n - p.n1;
    return p;
}

struct move_gain {
    float gain;
    docid_node* node;
    bool operator<(const move_gain& other) { return gain >= other.gain; }
};

struct move_gains_t {
    std::vector<move_gain> V1;
    std::vector<move_gain> V2;
};

move_gains_t compute_move_gains(std::vector<query_node>& Q, partition_t& P)
{
    move_gains_t G;
    for (const auto& q : Q) {
        // how does this work??
    }
    return G;
}

void recursive_bisection(
    std::vector<query_node>& Q, docid_node* G, size_t n, uint64_t depth = 0)
{
    // (1) create the initial partition
    auto partition = initial_partition(G, n);

    // (2) perform bisection
    for (uint64_t cur_iter = 1; cur_iter <= constants::MAX_ITERATIONS;
         cur_iter++) {
        // (2a) compute move gains. this has to happen in O(m) time,
        //  so O(1) per
        auto gains = compute_move_gains(Q, partition);

        // (2a) sort by decreasing gain. O(n log n)
        std::sort(gains.V1.begin(), gains.V1.end());
        std::sort(gains.V2.begin(), gains.V2.end());

        // (2b) swap. O(n)
        size_t num_swaps = 0;
        auto itr_v1 = gains.V1.begin();
        auto itr_v2 = gains.V2.begin();
        while (itr_v1 != gains.V1.end() && itr_v2 != gains.V2.end()) {
            if (itr_v1->gain + itr_v2->gain > 0) {
                // maybe we need to do something here to make
                // compute_move_gains() efficient?
                swap_nodes(itr_v1->node, itr_v2->node);
                num_swaps++;
            } else {
                break;
            }
            ++itr_v1;
            ++itr_v2;
        }

        // (2c) converged?
        if (num_swaps == 0) {
            break;
        }
    }

    // (4) recurse. at most O(log n) recursion steps
    if (depth + 1 <= constants::MAX_DEPTH) {
        if (partition.n1 > 1)
            recursive_bisection(Q, partition.V1, partition.n1, depth + 1);
        if (partition.n2 > 1)
            recursive_bisection(Q, partition.V2, partition.n2, depth + 1);
    }
}

docid_mapping reorder_docids_graph_bisection(inverted_index& invidx)
{
    auto bipartite_graph = construct_bipartite_graph(invidx);

    recursive_bisection(bipartite_graph.Q, bipartite_graph.G);

    return create_mapping(Q, G);
}
