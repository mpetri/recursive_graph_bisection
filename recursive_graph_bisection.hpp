#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "util.hpp"

namespace constants {
const uint64_t MAX_DEPTH = 13;
const uint64_t MAX_ITER = 10;
}

struct docid_node {
    uint64_t initial_id;
    uint32_t* terms;
    size_t num_terms;
};

struct bipartite_graph {
    size_t num_queries;
    std::vector<docid_node> graph;
    std::vector<uint32_t> doc_contents;
};

struct partition_t {
    docid_node* V1;
    docid_node* V2;
    size_t n1;
    size_t n2;
};

bipartite_graph construct_bipartite_graph(inverted_index& idx)
{
    timer t("construct_bipartite_graph");
    bipartite_graph bg;
    bg.num_queries = idx.size();
    uint32_t max_doc_id = 0;
    {
        size_t doc_size_sum = 0;
        std::vector<uint32_t> doc_sizes;
        std::cout << "determine doc sizes:" << std::endl;
        progress_bar progress(idx.size());
        for (size_t termid = 0; termid < idx.size(); termid++) {
            const auto& plist = idx[termid];
            for (const auto& doc_id : plist) {
                max_doc_id = std::max(max_doc_id, doc_id);
                if (doc_sizes.size() <= max_doc_id) {
                    doc_sizes.resize(1 + max_doc_id * 2);
                }
                doc_sizes[doc_id]++;
                doc_size_sum++;
            }
            ++progress;
        }
        bg.doc_contents.resize(doc_size_sum);
        doc_sizes.resize(max_doc_id + 1);
        bg.graph.resize(max_doc_id + 1);
        bg.graph[0].terms = bg.doc_contents.data();
        bg.graph[0].num_terms = doc_sizes[0];
        for (size_t i = 1; i < doc_sizes.size(); i++) {
            bg.graph[i].terms = bg.graph[i - 1].terms + bg.graph[i].num_terms;
            bg.graph[i].num_terms = doc_sizes[i];
        }
    }
    {
        std::cout << "creating forward index:" << std::endl;
        progress_bar progress(idx.size());
        std::vector<uint32_t> doc_offset(max_doc_id + 1, 0);
        for (size_t termid = 0; termid < idx.size(); termid++) {
            const auto& plist = idx[termid];
            for (const auto& doc_id : plist) {
                bg.graph[doc_id].initial_id = doc_id;
                bg.graph[doc_id].terms[doc_offset[doc_id]++] = termid;
            }
            ++progress;
        }
    }
    return bg;
}

inverted_index recreate_invidx(const bipartite_graph& bg)
{
    timer t("recreate_invidx");
    inverted_index idx;
    uint32_t max_qid_id = 0;
    for (size_t docid = 0; docid < bg.graph.size(); docid++) {
        const auto& doc = bg.graph[docid];
        for (size_t i = 0; i < doc.num_terms; i++) {
            auto qid = doc.terms[i];
            max_qid_id = std::max(max_qid_id, qid);
            if (idx.size() <= qid) {
                idx.resize(1 + qid * 2);
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
    timer t("initial_partition n=" + std::to_string(n));
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

move_gains_t compute_move_gains(partition_t& P, size_t num_queries)
{
    timer t("compute_move_gains n1=" + std::to_string(P.n1) + " n2="
        + std::to_string(P.n2));
    move_gains_t gains;

    // (1) compute current partition cost deg1/deg2
    std::vector<uint32_t> deg1(num_queries, 0);
    for (size_t i = 0; i < P.n1; i++) {
        auto doc = P.V1 + i;
        for (size_t j = 0; j < doc->num_terms; j++) {
            deg1[doc->terms[j]]++;
        }
    }
    std::vector<uint32_t> deg2(num_queries, 0);
    for (size_t i = 0; i < P.n2; i++) {
        auto doc = P.V2 + i;
        for (size_t j = 0; j < doc->num_terms; j++) {
            deg2[doc->terms[j]]++;
        }
    }

    // (2) compute gains from moving docs
    for (size_t i = 0; i < P.n1; i++) {
        auto doc = P.V1 + i;
        float before_move = 0.0;
        float after_move = 0.0;
        for (size_t j = 0; j < doc->num_terms; j++) {
            auto q = doc->terms[j];
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
        for (size_t j = 0; j < doc->num_terms; j++) {
            auto q = doc->terms[j];
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

void recursive_bisection(docid_node* G, size_t nq, size_t n, uint64_t depth = 0)
{
    // (1) create the initial partition. O(n)
    auto partition = initial_partition(G, n);

    // (2) perform bisection. constant number of iterations
    for (uint64_t cur_iter = 1; cur_iter <= constants::MAX_ITER; cur_iter++) {
        // (2a) compute move gains
        auto gains = compute_move_gains(partition, nq);

        // (2a) sort by decreasing gain. O(n log n)
        {
            timer t("sort by decreasing gain n=" + std::to_string(n));
            std::sort(gains.V1.begin(), gains.V1.end());
            std::sort(gains.V2.begin(), gains.V2.end());
        }
        // (2b) swap. O(n)
        size_t num_swaps = 0;
        {
            timer t("swap stuff");
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
        }

        // (2c) converged?
        if (num_swaps == 0) {
            break;
        }
    }

    // (3) recurse. at most O(log n) recursion steps
    if (depth + 1 <= constants::MAX_DEPTH) {
        timer t("recurse n=" + std::to_string(n));
        if (partition.n1 > 1)
            recursive_bisection(partition.V1, nq, partition.n1, depth + 1);
        if (partition.n2 > 1)
            recursive_bisection(partition.V2, nq, partition.n2, depth + 1);
    }
}

inverted_index reorder_docids_graph_bisection(inverted_index& invidx)
{
    std::cout << "construct_bipartite_graph" << std::endl;
    auto bg = construct_bipartite_graph(invidx);

    recursive_bisection(bg.graph.data(), bg.num_queries, bg.graph.size());

    std::cout << "recreate_invidx" << std::endl;
    return recreate_invidx(bg);
}
