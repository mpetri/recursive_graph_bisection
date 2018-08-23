#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "util.hpp"

#include <cilk/cilk.h>
#include <cilk/reducer_list.h>

namespace constants {
const uint64_t MAX_DEPTH = 15;
const int MAX_ITER = 20;
const uint64_t PARALLEL_SWITCH_DEPTH = 6;
}

struct docid_node {
    uint64_t initial_id;
    uint32_t* terms;
    uint32_t* freqs;
    size_t num_terms;
    size_t num_terms_not_pruned;
};

std::vector<float> log2_precomp;

void swap_nodes(docid_node* a, docid_node* b, std::vector<uint32_t>& deg1,
    std::vector<uint32_t>& deg2, std::vector<uint8_t>& queries_changed)
{
    for (size_t i = 0; i < a->num_terms; i++) {
        auto qry = a->terms[i];
        deg1[qry]--;
        deg2[qry]++;
        queries_changed[qry] = 1;
    }

    for (size_t i = 0; i < b->num_terms; i++) {
        auto qry = b->terms[i];
        deg1[qry]++;
        deg2[qry]--;
        queries_changed[qry] = 1;
    }

    std::swap(a->initial_id, b->initial_id);
    std::swap(a->terms, b->terms);
    std::swap(a->freqs, b->freqs);
    std::swap(a->num_terms, b->num_terms);
    std::swap(a->num_terms_not_pruned, b->num_terms_not_pruned);
}

void swap_nodes(docid_node* a, docid_node* b)
{
    std::swap(a->initial_id, b->initial_id);
    std::swap(a->terms, b->terms);
    std::swap(a->freqs, b->freqs);
    std::swap(a->num_terms, b->num_terms);
    std::swap(a->num_terms_not_pruned, b->num_terms_not_pruned);
}

struct bipartite_graph {
    size_t num_queries;
    size_t num_docs;
    size_t num_docs_inc_empty;
    std::vector<docid_node> graph;
    std::vector<uint32_t> doc_contents;
    std::vector<uint32_t> doc_freqs;
};

struct partition_t {
    docid_node* V1;
    docid_node* V2;
    size_t n1;
    size_t n2;
};

bipartite_graph construct_bipartite_graph(
    inverted_index& idx, size_t min_list_len)
{
    timer t("construct_bipartite_graph");
    bipartite_graph bg;
    bg.num_queries = idx.size();
    uint32_t max_doc_id = 0;
    {
        size_t doc_size_sum = 0;
        std::vector<uint32_t> doc_sizes;
        std::vector<uint32_t> doc_sizes_non_pruned;
        progress_bar progress("determine doc sizes", idx.size());
        for (size_t termid = 0; termid < idx.docids.size(); termid++) {
            const auto& plist = idx.docids[termid];
            for (const auto& doc_id : plist) {
                max_doc_id = std::max(max_doc_id, doc_id);
                if (doc_sizes.size() <= max_doc_id) {
                    doc_sizes.resize(1 + max_doc_id * 2);
                    doc_sizes_non_pruned.resize(1 + max_doc_id * 2);
                }
                if (plist.size() >= min_list_len) {
                    doc_sizes[doc_id]++;
                }
                doc_sizes_non_pruned[doc_id]++;
                doc_size_sum++;
            }
            ++progress;
        }
        bg.doc_contents.resize(doc_size_sum);
        bg.doc_freqs.resize(doc_size_sum);
        doc_sizes.resize(max_doc_id + 1);
        doc_sizes_non_pruned.resize(max_doc_id + 1);
        bg.graph.resize(max_doc_id + 1);
        bg.num_docs_inc_empty = max_doc_id + 1;
        bg.graph[0].terms = bg.doc_contents.data();
        bg.graph[0].freqs = bg.doc_freqs.data();
        bg.graph[0].num_terms = doc_sizes[0];
        bg.graph[0].num_terms_not_pruned = doc_sizes_non_pruned[0];
        for (size_t i = 1; i < doc_sizes.size(); i++) {
            bg.graph[i].terms
                = bg.graph[i - 1].terms + bg.graph[i - 1].num_terms_not_pruned;
            bg.graph[i].freqs
                = bg.graph[i - 1].freqs + bg.graph[i - 1].num_terms_not_pruned;
            bg.graph[i].num_terms = doc_sizes[i];
            bg.graph[i].num_terms_not_pruned = doc_sizes_non_pruned[i];
        }
    }
    {
        progress_bar progress("creating forward index", idx.size());
        std::vector<uint32_t> doc_offset(max_doc_id + 1, 0);
        for (size_t termid = 0; termid < idx.docids.size(); termid++) {
            const auto& dlist = idx.docids[termid];
            const auto& flist = idx.freqs[termid];
            if (dlist.size() >= min_list_len) {
                for (size_t pos = 0; pos < dlist.size(); pos++) {
                    const auto& doc_id = dlist[pos];
                    bg.graph[doc_id].initial_id = doc_id;
                    bg.graph[doc_id].freqs[doc_offset[doc_id]] = flist[pos];
                    bg.graph[doc_id].terms[doc_offset[doc_id]++] = termid;
                }
            }
            ++progress;
        }
    }
    // Set ID for empty documents.
    for (uint32_t doc_id = 0; doc_id < idx.num_docs; ++doc_id) {
        if (bg.graph[doc_id].initial_id != doc_id) {
            bg.graph[doc_id].initial_id = doc_id;
        }
    }
    {
        // all docs with 0 size go to the back!
        auto ritr = bg.graph.end() - 1;
        auto itr = bg.graph.begin();
        size_t num_empty = 0;
        while (itr != ritr) {
            if (itr->num_terms == 0) {
                // Find next non-empty doc from end
                while (ritr->num_terms == 0 && ritr != itr) {
                    num_empty++;
                    --ritr;
                }
                // Ensure we did not meet itr
                if (itr == ritr) {
                    break;
                }
                num_empty++;
                swap_nodes(&*itr, &*ritr);
                --ritr;
            }
            ++itr;
        }
        bg.num_docs = bg.num_docs_inc_empty - num_empty;
    }
    return bg;
}

inverted_index recreate_invidx(const bipartite_graph& bg)
{
    timer t("recreate_invidx");
    inverted_index idx;
    uint32_t max_qid_id = 0;
    progress_bar progress("recreate invidx", bg.num_docs_inc_empty);
    for (size_t docid = 0; docid < bg.num_docs_inc_empty; docid++) {
        uint32_t length_accumulator = 0;
        const auto& doc = bg.graph[docid];
        idx.doc_id_mapping.push_back(doc.initial_id);
        for (size_t i = 0; i < doc.num_terms_not_pruned; i++) {
            auto qid = doc.terms[i];
            auto freq = doc.freqs[i];
            max_qid_id = std::max(max_qid_id, qid);
            if (idx.size() <= qid) {
                idx.resize(1 + qid * 2);
            }
            idx.docids[qid].push_back(docid);
            idx.freqs[qid].push_back(freq);
            length_accumulator += freq;
        }
        idx.doc_lengths.push_back(length_accumulator);
        ++progress;
    }
    idx.num_docs = bg.num_docs_inc_empty;
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
    double gain;
    docid_node* node;
    move_gain()
        : gain(0)
        , node(nullptr)
    {
    }
    move_gain(double g, docid_node* n)
        : gain(g)
        , node(n)
    {
    }
    bool operator<(const move_gain& other) { return gain > other.gain; }
};

struct move_gains_t {
    std::vector<move_gain> V1;
    std::vector<move_gain> V2;
};

move_gain compute_single_gain(
    docid_node* doc, std::vector<float>& before, std::vector<float>& after)
{
    float before_move = 0.0;
    float after_move = 0.0;
    for (size_t j = 0; j < doc->num_terms; j++) {
        auto q = doc->terms[j];
        before_move += before[q];
        after_move += after[q];
    }
    float gain = before_move - after_move;
    return move_gain(gain, doc);
}

void compute_deg(docid_node* docs, size_t n, std::vector<uint32_t>& deg)
{
    for (size_t i = 0; i < n; i++) {
        auto doc = docs + i;
        for (size_t j = 0; j < doc->num_terms; j++) {
            deg[doc->terms[j]]++;
        }
    }
}

void compute_gains(docid_node* docs, size_t n, std::vector<float>& before,
    std::vector<float>& after, std::vector<move_gain>& res)
{
    cilk::reducer<cilk::op_list_append<move_gain> > gr;
    res.resize(n);
    cilk_for(size_t i = 0; i < n; i++)
    {
        auto doc = docs + i;
        res[i] = compute_single_gain(doc, before, after);
    }
}

void compute_gains_np(docid_node* docs, size_t n, std::vector<float>& before,
    std::vector<float>& after, std::vector<move_gain>& res)
{
    cilk::reducer<cilk::op_list_append<move_gain> > gr;
    res.resize(n);
    for (size_t i = 0; i < n; i++) {
        auto doc = docs + i;
        res[i] = compute_single_gain(doc, before, after);
    }
}

move_gains_t compute_move_gains(partition_t& P, size_t num_queries,
    std::vector<uint32_t>& deg1, std::vector<uint32_t>& deg2,
    std::vector<float>& before, std::vector<float>& left2right,
    std::vector<float>& right2left, std::vector<uint8_t>& qry_changed)
{
    move_gains_t gains;

    float logn1 = log2f(P.n1);
    float logn2 = log2f(P.n2);
    cilk_for(size_t q = 0; q < num_queries; q++)
    {
        if (qry_changed[q] == 1) {
            qry_changed[q] = 0;
            before[q] = 0;
            left2right[q] = 0;
            right2left[q] = 0;
            if (deg1[q] or deg2[q]) {
                before[q] = deg1[q] * logn1
                    - deg1[q] * log2_precomp[deg1[q] + 1] + deg2[q] * logn2
                    - deg2[q] * log2_precomp[deg2[q] + 1];
            }
            if (deg1[q]) {
                left2right[q] = (deg1[q] - 1) * logn1
                    - (deg1[q] - 1) * log2_precomp[deg1[q]]
                    + (deg2[q] + 1) * logn2
                    - (deg2[q] + 1) * log2_precomp[deg2[q] + 2];
            }
            if (deg2[q])
                right2left[q] = (deg1[q] + 1) * logn1
                    - (deg1[q] + 1) * log2_precomp[deg1[q] + 2]
                    + (deg2[q] - 1) * logn2
                    - (deg2[q] - 1) * log2_precomp[deg2[q]];
        }
    }

    // (2) compute gains from moving docs
    cilk_spawn compute_gains(P.V1, P.n1, before, left2right, gains.V1);
    cilk_spawn compute_gains(P.V2, P.n2, before, right2left, gains.V2);
    cilk_sync;

    return gains;
}

move_gains_t compute_move_gains_np(partition_t& P, size_t num_queries,
    std::vector<uint32_t>& deg1, std::vector<uint32_t>& deg2,
    std::vector<float>& before, std::vector<float>& left2right,
    std::vector<float>& right2left, std::vector<uint8_t>& qry_changed)
{
    move_gains_t gains;

    float logn1 = log2f(P.n1);
    float logn2 = log2f(P.n2);
    for (size_t q = 0; q < num_queries; q++) {
        if (qry_changed[q] == 1) {
            qry_changed[q] = 0;
            before[q] = 0;
            left2right[q] = 0;
            right2left[q] = 0;
            if (deg1[q] or deg2[q]) {
                before[q] = deg1[q] * logn1
                    - deg1[q] * log2_precomp[deg1[q] + 1] + deg2[q] * logn2
                    - deg2[q] * log2_precomp[deg2[q] + 1];
            }
            if (deg1[q]) {
                left2right[q] = (deg1[q] - 1) * logn1
                    - (deg1[q] - 1) * log2_precomp[deg1[q]]
                    + (deg2[q] + 1) * logn2
                    - (deg2[q] + 1) * log2_precomp[deg2[q] + 2];
            }
            if (deg2[q])
                right2left[q] = (deg1[q] + 1) * logn1
                    - (deg1[q] + 1) * log2_precomp[deg1[q] + 2]
                    + (deg2[q] - 1) * logn2
                    - (deg2[q] - 1) * log2_precomp[deg2[q]];
        }
    }

    // (2) compute gains from moving docs
    compute_gains(P.V1, P.n1, before, left2right, gains.V1);
    compute_gains(P.V2, P.n2, before, right2left, gains.V2);

    return gains;
}

void recursive_bisection_np(progress_bar& progress, docid_node* G,
    size_t num_queries, size_t n, uint64_t depth = 0)
{
    // (1) create the initial partition. O(n)
    auto partition = initial_partition(G, n);

    {
        // (2) we compute deg1 and deg2 only once
        std::vector<uint32_t> deg1(num_queries, 0);
        std::vector<uint32_t> deg2(num_queries, 0);
        std::vector<float> before(num_queries);
        std::vector<float> left2right(num_queries);
        std::vector<float> right2left(num_queries);

        std::vector<uint8_t> query_changed(num_queries, 1);
        {
            compute_deg(partition.V1, partition.n1, deg1);
            compute_deg(partition.V2, partition.n2, deg2);
        }

        // (3) perform bisection. constant number of iterations
        for (int cur_iter = 1; cur_iter <= constants::MAX_ITER; cur_iter++) {
            // (3a) compute move gains
            auto gains = compute_move_gains_np(partition, num_queries, deg1,
                deg2, before, left2right, right2left, query_changed);
            memset(query_changed.data(), 0, num_queries);

            // (3b) sort by decreasing gain. O(n log n)
            {
                std::sort(gains.V1.begin(), gains.V1.end());
                std::sort(gains.V2.begin(), gains.V2.end());
            }

            // (3c) swap. O(n)
            size_t num_swaps = 0;
            {
                auto itr_v1 = gains.V1.begin();
                auto itr_v2 = gains.V2.begin();
                while (itr_v1 != gains.V1.end() && itr_v2 != gains.V2.end()) {
                    if (itr_v1->gain + itr_v2->gain > 0) {
                        // maybe we need to do something here to make
                        // compute_move_gains() efficient?
                        swap_nodes(itr_v1->node, itr_v2->node, deg1, deg2,
                            query_changed);
                        num_swaps++;
                    } else {
                        break;
                    }
                    ++itr_v1;
                    ++itr_v2;
                }
            }

            // (3d) converged?
            if (num_swaps == 0) {
                break;
            }
        }
    }

    // (4) recurse. at most O(log n) recursion steps
    if (depth + 1 <= constants::MAX_DEPTH) {
        if (partition.n1 > 1)
            recursive_bisection_np(
                progress, partition.V1, num_queries, partition.n1, depth + 1);
        if (partition.n2 > 1)
            recursive_bisection_np(
                progress, partition.V2, num_queries, partition.n2, depth + 1);

        if (partition.n1 == 1)
            progress.done(1);
        if (partition.n2 == 1)
            progress.done(1);
    } else {
        progress.done(n);
    }
}

void recursive_bisection(progress_bar& progress, docid_node* G,
    size_t num_queries, size_t n, uint64_t depth = 0)
{
    // (1) create the initial partition. O(n)
    auto partition = initial_partition(G, n);

    {
        // (2) we compute deg1 and deg2 only once
        std::vector<uint32_t> deg1(num_queries, 0);
        std::vector<uint32_t> deg2(num_queries, 0);
        std::vector<float> before(num_queries);
        std::vector<float> left2right(num_queries);
        std::vector<float> right2left(num_queries);

        std::vector<uint8_t> query_changed(num_queries, 1);
        {
            cilk_spawn compute_deg(partition.V1, partition.n1, deg1);
            cilk_spawn compute_deg(partition.V2, partition.n2, deg2);
            cilk_sync;
        }

        // (3) perform bisection. constant number of iterations
        for (int cur_iter = 1; cur_iter <= constants::MAX_ITER; cur_iter++) {
            // (3a) compute move gains
            auto gains = compute_move_gains(partition, num_queries, deg1, deg2,
                before, left2right, right2left, query_changed);
            memset(query_changed.data(), 0, num_queries);

            // (3b) sort by decreasing gain. O(n log n)
            {
                cilk_spawn std::sort(gains.V1.begin(), gains.V1.end());
                cilk_spawn std::sort(gains.V2.begin(), gains.V2.end());
                cilk_sync;
            }

            // (3c) swap. O(n)
            size_t num_swaps = 0;
            {
                auto itr_v1 = gains.V1.begin();
                auto itr_v2 = gains.V2.begin();
                while (itr_v1 != gains.V1.end() && itr_v2 != gains.V2.end()) {
                    if (itr_v1->gain + itr_v2->gain > 0) {
                        // maybe we need to do something here to make
                        // compute_move_gains() efficient?
                        swap_nodes(itr_v1->node, itr_v2->node, deg1, deg2,
                            query_changed);
                        num_swaps++;
                    } else {
                        break;
                    }
                    ++itr_v1;
                    ++itr_v2;
                }
            }

            // (3d) converged?
            if (num_swaps == 0) {
                break;
            }
        }
    }

    // (4) recurse. at most O(log n) recursion steps
    if (depth + 1 <= constants::MAX_DEPTH) {
        if (depth < constants::PARALLEL_SWITCH_DEPTH) {
            if (partition.n1 > 1) {
                cilk_spawn recursive_bisection(progress, partition.V1,
                    num_queries, partition.n1, depth + 1);
            }
            if (partition.n2 > 1) {
                cilk_spawn recursive_bisection(progress, partition.V2,
                    num_queries, partition.n2, depth + 1);
            }
            cilk_sync;
        } else {
            if (partition.n1 > 1) {
                recursive_bisection_np(progress, partition.V1, num_queries,
                    partition.n1, depth + 1);
            }
            if (partition.n2 > 1) {
                recursive_bisection_np(progress, partition.V2, num_queries,
                    partition.n2, depth + 1);
            }
        }
        if (partition.n1 == 1)
            progress.done(1);
        if (partition.n2 == 1)
            progress.done(1);
    } else {
        progress.done(n);
    }
}

inverted_index reorder_docids_graph_bisection(
    inverted_index& invidx, size_t min_list_len)
{
    auto bg = construct_bipartite_graph(invidx, min_list_len);

    // free up some space
    invidx.clear();

    // make things faster by precomputing some logs
    log2_precomp.resize(bg.num_docs);
    cilk_for(size_t i = 0; i < bg.num_docs; i++) { log2_precomp[i] = log2f(i); }

    {
        timer t("recursive_bisection");
        progress_bar bp("recursive_bisection", bg.num_docs);
        recursive_bisection(bp, bg.graph.data(), bg.num_queries, bg.num_docs);
    }
    return recreate_invidx(bg);
}
