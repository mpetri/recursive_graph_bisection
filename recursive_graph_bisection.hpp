#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "util.hpp"

namespace constants {
const uint64_t MAX_DEPTH = 13;
const uint64_t MAX_ITER = 10;
}

struct docid_node {
    uint64_t initial_id;
    std::vector<uint32_t> terms;
};

using bipartite_graph = std::vector<docid_node>;

struct partition_t {
    docid_node* V1;
    docid_node* V2;
    size_t n1;
    size_t n2;
};

bipartite_graph construct_bipartite_graph(inverted_index& idx)
{
    bipartite_graph bg;
    uint32_t max_doc_id = 0;
    for (size_t termid = 0; termid < idx.size(); termid++) {
        const auto& plist = idx[termid];
        for (const auto& doc_id : plist) {
            max_doc_id = std::max(max_doc_id, doc_id);
            if (bg.size() <= doc_id) {
                bg.resize(bg.size() * 2);
            }
            bg[doc_id].initial_id = doc_id;
            bg[doc_id].terms.push_back(termid);
        }
    }
    bg.resize(max_doc_id + 1);
    return bg;
}

inverted_index recreate_invidx(const bipartite_graph& bg)
{
    inverted_index idx;
    uint32_t max_qid_id = 0;
    for (size_t docid = 0; docid < bg.size(); docid++) {
        const auto& doc = bg[docid];
        for (const auto& qid : doc.terms) {
            max_qid_id = std::max(max_qid_id, qid);
            if (idx.size() <= qid) {
                idx.resize(idx.size() * 2);
            }
            idx[qid].push_back(docid);
        }
    }
    idx.resize(max_qid_id + 1);
    return idx;
}

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
    move_gain(float g, docid_node* n)
        : gain(g)
        , node(n)
    {
    }
    bool operator<(const move_gain& other) { return gain >= other.gain; }
};

struct move_gains_t {
    std::vector<move_gain> V1;
    std::vector<move_gain> V2;
};

move_gains_t compute_move_gains(partition_t& P)
{
    move_gains_t gains;

    // (1) compute current partition cost deg1/deg2
    std::unordered_map<uint32_t, uint32_t> deg1;
    for (size_t i = 0; i < P.n1; i++) {
        auto doc = P.V1 + i;
        for (const auto& q : doc->terms) {
            deg1[q]++;
        }
    }
    std::unordered_map<uint32_t, uint32_t> deg2;
    for (size_t i = 0; i < P.n2; i++) {
        auto doc = P.V2 + i;
        for (const auto& q : doc->terms) {
            deg2[q]++;
        }
    }

    // (2) compute gains from moving docs
    for (size_t i = 0; i < P.n1; i++) {
        auto doc = P.V1 + i;
        float before_move = 0.0;
        float after_move = 0.0;
        for (const auto& q : doc->terms) {
            float d1 = deg1[q];
            float d2 = deg2[q];
            before_move += ((d1 * logf(float(P.n1) / (d1 + 1)))
                + (d2 * logf(float(P.n2) / (d2 + 1))));
            after_move += (((d1 - 1) * logf(float(P.n1) / ((d1 - 1) + 1)))
                + ((d2 + 1) * logf(float(P.n2) / ((d2 + 1) + 1))));
        }
        float gain = before_move - after_move;
        gains.V1.emplace_back(gain, doc);
    }

    for (size_t i = 0; i < P.n2; i++) {
        auto doc = P.V2 + i;
        float before_move = 0.0;
        float after_move = 0.0;
        for (const auto& q : doc->terms) {
            float d1 = deg1[q];
            float d2 = deg2[q];
            before_move += ((d1 * logf(float(P.n1) / (d1 + 1)))
                + (d2 * logf(float(P.n2) / (d2 + 1))));
            after_move += (((d1 + 1) * logf(float(P.n1) / ((d1 + 1) + 1)))
                + ((d2 - 1) * logf(float(P.n2) / ((d2 - 1) + 1))));
        }
        float gain = before_move - after_move;
        gains.V2.emplace_back(gain, doc);
    }

    return gains;
}

void swap_nodes(docid_node* a, docid_node* b)
{
    std::swap(a->initial_id, b->initial_id);
    std::swap(a->terms, b->terms);
}

void recursive_bisection(docid_node* G, size_t n, uint64_t depth = 0)
{
    // (1) create the initial partition. O(n)
    std::cout << "initial_partition n=" << n << std::endl;
    auto partition = initial_partition(G, n);

    // (2) perform bisection. constant number of iterations
    for (uint64_t cur_iter = 1; cur_iter <= constants::MAX_ITER; cur_iter++) {
        // (2a) compute move gains
        std::cout << "compute_move_gains n=" << n << std::endl;
        auto gains = compute_move_gains(partition);

        // (2a) sort by decreasing gain. O(n log n)
        std::cout << "sort by decreasing gain n=" << n << std::endl;
        std::sort(gains.V1.begin(), gains.V1.end());
        std::sort(gains.V2.begin(), gains.V2.end());

        // (2b) swap. O(n)
        std::cout << "swap n=" << n << std::endl;
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

    // (3) recurse. at most O(log n) recursion steps
    std::cout << "recurse n=" << n << std::endl;
    if (depth + 1 <= constants::MAX_DEPTH) {
        if (partition.n1 > 1)
            recursive_bisection(partition.V1, partition.n1, depth + 1);
        if (partition.n2 > 1)
            recursive_bisection(partition.V2, partition.n2, depth + 1);
    }
}

inverted_index reorder_docids_graph_bisection(inverted_index& invidx)
{
    std::cout << "construct_bipartite_graph" << std::endl;
    auto bipartite_graph = construct_bipartite_graph(invidx);

    recursive_bisection(bipartite_graph.data(), bipartite_graph.size());

    std::cout << "recreate_invidx" << std::endl;
    return recreate_invidx(bipartite_graph);
}
