#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <x86intrin.h>

#include "util.hpp"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_max.h>
#include <cilk/reducer_opadd.h>

namespace constants {
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

float log2_cmp(uint32_t idx)
{
    if (idx < 256)
        return log2_precomp[idx];
    return log2f(idx);
}

void swap_nodes(docid_node* a, docid_node* b)
{
    std::swap(a->initial_id, b->initial_id);
    std::swap(a->terms, b->terms);
    std::swap(a->freqs, b->freqs);
    std::swap(a->num_terms, b->num_terms);
    std::swap(a->num_terms_not_pruned, b->num_terms_not_pruned);
}

void swap_nodes(docid_node* a, docid_node* b, std::vector<uint32_t>& deg1,
    std::vector<uint32_t>& deg2, std::vector<uint8_t>& queries_changed)
{
    {
        size_t n = a->num_terms / 4;
        size_t m = a->num_terms % 4;
        for (size_t i = 0; i < n * 4; i+=4) {
            auto q0 = a->terms[i];
            auto q1 = a->terms[i + 1];
            auto q2 = a->terms[i + 2];
            auto q3 = a->terms[i + 3];
            __m128i _one = _mm_set1_epi32(1);

            {
                __m128i _deg1 = _mm_set_epi32(deg1[q0], deg1[q1], deg1[q2], deg1[q3]);
                __m128i _result = _mm_sub_epi32(_deg1, _one);
                deg1[q0] = _mm_extract_epi32(_result, 3);
                deg1[q1] = _mm_extract_epi32(_result, 2);
                deg1[q2] = _mm_extract_epi32(_result, 1);
                deg1[q3] = _mm_extract_epi32(_result, 0);
            }
            {
                __m128i _deg2 = _mm_set_epi32(deg2[q0], deg2[q1], deg2[q2], deg2[q3]);
                __m128i _result = _mm_add_epi32(_deg2, _one);
                deg2[q0] = _mm_extract_epi32(_result, 3);
                deg2[q1] = _mm_extract_epi32(_result, 2);
                deg2[q2] = _mm_extract_epi32(_result, 1);
                deg2[q3] = _mm_extract_epi32(_result, 0);

            }
            queries_changed[q0] = 1;
            queries_changed[q1] = 1;
            queries_changed[q2] = 1;
            queries_changed[q3] = 1;
        }
        for (size_t i = 0; i < m; i++) {
            auto qry = a->terms[n * 4 + i];
            deg1[qry]--;
            deg2[qry]++;
            queries_changed[qry] = 1;
        }
    }
    {
        size_t n = b->num_terms / 4;
        size_t m = b->num_terms % 4;
        for (size_t i = 0; i < n * 4; i+=4) {
            auto q0 = b->terms[i];
            auto q1 = b->terms[i + 1];
            auto q2 = b->terms[i + 2];
            auto q3 = b->terms[i + 3];
            __m128i _one = _mm_set1_epi32(1);

            {
                __m128i _deg1 = _mm_set_epi32(deg1[q0], deg1[q1], deg1[q2], deg1[q3]);
                __m128i _result = _mm_add_epi32(_deg1, _one);
                deg1[q0] = _mm_extract_epi32(_result, 3);
                deg1[q1] = _mm_extract_epi32(_result, 2);
                deg1[q2] = _mm_extract_epi32(_result, 1);
                deg1[q3] = _mm_extract_epi32(_result, 0);
            }
            {
                __m128i _deg2 = _mm_set_epi32(deg2[q0], deg2[q1], deg2[q2], deg2[q3]);
                __m128i _result = _mm_sub_epi32(_deg2, _one);
                deg2[q0] = _mm_extract_epi32(_result, 3);
                deg2[q1] = _mm_extract_epi32(_result, 2);
                deg2[q2] = _mm_extract_epi32(_result, 1);
                deg2[q3] = _mm_extract_epi32(_result, 0);

            }
            queries_changed[q0] = 1;
            queries_changed[q1] = 1;
            queries_changed[q2] = 1;
            queries_changed[q3] = 1;
        }
        for (size_t i = 0; i < m; i++) {
            auto qry = b->terms[n * 4 + i];
            deg1[qry]++;
            deg2[qry]--;
            queries_changed[qry] = 1;
        }
    }
    swap_nodes(a, b);
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

void compute_doc_sizes(inverted_index& idx, std::vector<uint32_t>& doc_sizes,
    std::vector<uint32_t>& doc_sizes_non_pruned, uint32_t min_doc_id,
    uint32_t max_doc_id, size_t min_list_len)
{
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& plist = idx.docids[termid];
        for (const auto& doc_id : plist) {
            if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                if (plist.size() >= min_list_len) {
                    doc_sizes[doc_id] += 1;
                }
                doc_sizes_non_pruned[doc_id] += 1;
            }
        }
    }
}

void create_graph(bipartite_graph& bg, inverted_index& idx, uint32_t min_doc_id,
    uint32_t max_doc_id, size_t min_list_len)
{
    std::vector<uint32_t> doc_offset(idx.max_doc_id + 1, 0);
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& dlist = idx.docids[termid];
        const auto& flist = idx.freqs[termid];
        if (dlist.size() >= min_list_len) {
            for (size_t pos = 0; pos < dlist.size(); pos++) {
                const auto& doc_id = dlist[pos];
                if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                    bg.graph[doc_id].initial_id = doc_id;
                    bg.graph[doc_id].freqs[doc_offset[doc_id]] = flist[pos];
                    bg.graph[doc_id].terms[doc_offset[doc_id]++] = termid;
                }
            }
        }
    }
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& dlist = idx.docids[termid];
        const auto& flist = idx.freqs[termid];
        if (dlist.size() < min_list_len) {
            for (size_t pos = 0; pos < dlist.size(); pos++) {
                const auto& doc_id = dlist[pos];
                if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                    bg.graph[doc_id].initial_id = doc_id;
                    bg.graph[doc_id].freqs[doc_offset[doc_id]] = flist[pos];
                    bg.graph[doc_id].terms[doc_offset[doc_id]++] = termid;
                }
            }
        }
    }
}

bipartite_graph construct_bipartite_graph(
    inverted_index& idx, size_t min_list_len)
{
    timer t("construct_bipartite_graph");
    bipartite_graph bg;
    bg.num_queries = idx.size();
    {
        timer t("determine doc sizes");
        size_t workers = __cilkrts_get_nworkers();
        std::vector<uint32_t> doc_sizes(idx.max_doc_id + 1);
        std::vector<uint32_t> doc_sizes_non_pruned(idx.max_doc_id + 1);
        std::vector<std::vector<uint32_t>> tmp_doc_sizes(workers);
        std::vector<std::vector<uint32_t>> tmp_doc_sizes_non_pruned(workers);
        for (auto& v : tmp_doc_sizes)
            v.resize(idx.max_doc_id + 1);
        for (auto& v : tmp_doc_sizes_non_pruned)
            v.resize(idx.max_doc_id + 1);
        size_t doc_ids_in_slice = idx.max_doc_id / workers;
        for (size_t id = 0; id < workers; id++) {
            size_t min_doc_id = id * doc_ids_in_slice;
            size_t max_doc_id = min_doc_id + doc_ids_in_slice;
            if (id + 1 == workers) {
                max_doc_id = idx.max_doc_id + 1;
                compute_doc_sizes(idx, tmp_doc_sizes[id],
                    tmp_doc_sizes_non_pruned[id], min_doc_id, max_doc_id,
                    min_list_len);
            } else {
                cilk_spawn compute_doc_sizes(idx, tmp_doc_sizes[id],
                    tmp_doc_sizes_non_pruned[id], min_doc_id, max_doc_id,
                    min_list_len);
            }
        }
        cilk_sync;
        for (auto& v : tmp_doc_sizes) {
            for (size_t i = 0; i < v.size(); i++) {
                if (v[i] != 0)
                    doc_sizes[i] = v[i];
            }
        }
        for (auto& v : tmp_doc_sizes_non_pruned) {
            for (size_t i = 0; i < v.size(); i++) {
                if (v[i] != 0)
                    doc_sizes_non_pruned[i] = v[i];
            }
        }
        bg.doc_contents.resize(idx.num_postings);
        bg.doc_freqs.resize(idx.num_postings);
        bg.graph.resize(idx.max_doc_id + 1);
        bg.num_docs_inc_empty = idx.max_doc_id + 1;
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
        timer t("create forward index");
        size_t workers = __cilkrts_get_nworkers();
        size_t doc_ids_in_slice = idx.max_doc_id / workers;
        for (size_t id = 0; id < workers; id++) {
            size_t min_doc_id = id * doc_ids_in_slice;
            size_t max_doc_id = min_doc_id + doc_ids_in_slice;
            if (id + 1 == workers) {
                max_doc_id = idx.max_doc_id + 1;
                create_graph(bg, idx, min_doc_id, max_doc_id, min_list_len);
            } else {
                cilk_spawn create_graph(
                    bg, idx, min_doc_id, max_doc_id, min_list_len);
            }
        }
        cilk_sync;
    }

    // Set ID for empty documents.
    for (uint32_t doc_id = 0; doc_id < idx.num_docs; ++doc_id) {
        if (bg.graph[doc_id].initial_id != doc_id) {
            bg.graph[doc_id].initial_id = doc_id;
        }
    }
    size_t num_empty = 0;
    {
        // all docs with 0 size go to the back!
        auto empty_cmp = [](const auto& a, const auto& b) {
            return a.num_terms > b.num_terms;
        };
        std::sort(bg.graph.begin(), bg.graph.end(), empty_cmp);
        auto ritr = bg.graph.end() - 1;
        auto itr = bg.graph.begin();
        while (itr != ritr) {
            if (itr->num_terms == 0) {
                num_empty++;
            } else {
                break;
            }
            --ritr;
        }
        bg.num_docs = bg.num_docs_inc_empty - num_empty;
    }

    size_t num_skipped_lists = 0;
    size_t num_lists = 0;
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& dlist = idx.docids[termid];
        if (dlist.size() < min_list_len) {
            num_skipped_lists++;
        } else {
            num_lists++;
        }
    }
    std::cout << "\tnum_empty docs = " << num_empty << std::endl;
    std::cout << "\tnum_skipped lists = " << num_skipped_lists << std::endl;
    std::cout << "\tnum_lists = " << num_lists << std::endl;
    std::cout << "\tnum_docs = " << bg.num_docs << std::endl;
    return bg;
}

void recreate_lists(const bipartite_graph& bg, inverted_index& idx,
    uint32_t min_q_id, uint32_t max_q_id, std::vector<uint32_t>& qmap,
    std::vector<uint32_t>& dsizes)
{
    for (size_t docid = 0; docid < bg.num_docs_inc_empty; docid++) {
        const auto& doc = bg.graph[docid];
        for (size_t i = 0; i < doc.num_terms_not_pruned; i++) {
            auto qid = doc.terms[i];
            if (min_q_id <= qmap[qid] && qmap[qid] < max_q_id) {
                auto freq = doc.freqs[i];
                idx.docids[qid].push_back(docid);
                idx.freqs[qid].push_back(freq);
                dsizes[docid] += freq;
            }
        }
    }
}

inverted_index recreate_invidx(const bipartite_graph& bg, size_t num_lists)
{
    timer t("recreate_invidx");
    inverted_index idx;
    size_t num_postings = 0;
    idx.resize(num_lists);
    {
        size_t workers = __cilkrts_get_nworkers();
        size_t qids_in_slice = num_lists / workers;
        std::vector<uint32_t> qids_map(num_lists);
        for (size_t i = 0; i < qids_map.size(); i++)
            qids_map[i] = i;
        std::mt19937 rnd(1);
        std::shuffle(qids_map.begin(), qids_map.end(), rnd);
        std::vector<std::vector<uint32_t>> doc_sizes(workers);
        for (size_t id = 0; id < workers; id++) {
            doc_sizes[id].resize(bg.num_docs_inc_empty);
            size_t min_q_id = id * qids_in_slice;
            size_t max_q_id = min_q_id + qids_in_slice;
            if (id + 1 == workers) {
                max_q_id = num_lists;
                recreate_lists(
                    bg, idx, min_q_id, max_q_id, qids_map, doc_sizes[id]);
            } else {
                cilk_spawn recreate_lists(
                    bg, idx, min_q_id, max_q_id, qids_map, doc_sizes[id]);
            }
        }
        cilk_sync;
        idx.doc_lengths.resize(bg.num_docs_inc_empty);
        for (size_t id = 0; id < workers; id++) {
            for (size_t docid = 0; docid < bg.num_docs_inc_empty; docid++) {
                idx.doc_lengths[docid] += doc_sizes[id][docid];
            }
        }
    }
    {

        for (size_t docid = 0; docid < bg.num_docs_inc_empty; docid++) {
            const auto& doc = bg.graph[docid];
            idx.doc_id_mapping.push_back(doc.initial_id);
            num_postings += doc.num_terms_not_pruned;
        }
    }
    idx.num_docs = bg.num_docs_inc_empty;
    idx.max_doc_id = idx.num_docs - 1;
    idx.num_postings = num_postings;
    std::cout << "\tnum_docs = " << idx.num_docs << std::endl;
    std::cout << "\tmax_doc_id = " << idx.max_doc_id << std::endl;
    std::cout << "\tnum_lists = " << idx.docids.size() << std::endl;
    std::cout << "\tnum_postings = " << idx.num_postings << std::endl;
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

move_gain compute_single_gain(docid_node* doc,
    std::vector<float>& before, std::vector<float>& after)
{
    __m128 _vsum = _mm_set1_ps(0);
    float gain[4];
    size_t n = doc->num_terms / 4;
    size_t m = doc->num_terms % 4;
    for (size_t j = 0; j < n * 4; j+=4) {
        auto q0 = doc->terms[j];
        auto q1 = doc->terms[j + 1];
        auto q2 = doc->terms[j + 2];
        auto q3 = doc->terms[j + 3];
        __m128 _before = _mm_set_ps(before[q0], before[q1], before[q2], before[q3]);
        __m128 _after = _mm_set_ps(after[q0], after[q1], after[q2], after[q3]);
        __m128 _val = _mm_sub_ps(_before, _after);
        _vsum = _mm_add_ps(_vsum, _val);
    }
    _mm_store_ps(gain, _vsum);
    auto total = gain[0] + gain[1] + gain[2] + gain[3];
    for (size_t j = 0; j < m; j++) {
        auto q = doc->terms[n * 4 + j];
        total += before[q] - after[q];
    }
    return move_gain(total, doc);
}

void compute_deg(docid_node* docs, size_t n, std::vector<uint32_t>& deg, std::vector<uint8_t> &query_changed)
{
    for (size_t i = 0; i < n; i++) {
        auto doc = docs + i;
        size_t n = doc->num_terms / 4;
        size_t m = doc->num_terms % 4;
        for (size_t j = 0; j < n * 4; j+=4) {
            auto q0 = doc->terms[j];
            auto q1 = doc->terms[j + 1];
            auto q2 = doc->terms[j + 2];
            auto q3 = doc->terms[j + 3];
            __m128i _one = _mm_set1_epi32(1);
            __m128i _deg = _mm_set_epi32(deg[q0], deg[q1], deg[q2], deg[q3]);
            __m128i _result = _mm_add_epi32(_deg, _one);
            deg[q0] = _mm_extract_epi32(_result, 3);
            deg[q1] = _mm_extract_epi32(_result, 2);
            deg[q2] = _mm_extract_epi32(_result, 1);
            deg[q3] = _mm_extract_epi32(_result, 0);

            query_changed[q0] = 1;
            query_changed[q1] = 1;
            query_changed[q2] = 1;
            query_changed[q3] = 1;
        }
        for (size_t j = 0; j < m; j++) {
            auto qry = doc->terms[n * 4 + j];
            deg[qry]++;
            query_changed[qry] = 1;
        }
    }
}

void compute_gains(docid_node* docs, size_t n, std::vector<float>& before,
    std::vector<float>& after, std::vector<move_gain>& res)
{
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
                before[q] = deg1[q] * logn1 - deg1[q] * log2_cmp(deg1[q] + 1)
                    + deg2[q] * logn2 - deg2[q] * log2_cmp(deg2[q] + 1);
            }
            if (deg1[q]) {
                left2right[q] = (deg1[q] - 1) * logn1
                    - (deg1[q] - 1) * log2_cmp(deg1[q]) + (deg2[q] + 1) * logn2
                    - (deg2[q] + 1) * log2_cmp(deg2[q] + 2);
            }
            if (deg2[q])
                right2left[q] = (deg1[q] + 1) * logn1
                    - (deg1[q] + 1) * log2_cmp(deg1[q] + 2)
                    + (deg2[q] - 1) * logn2 - (deg2[q] - 1) * log2_cmp(deg2[q]);
        }
    }

    // (2) compute gains from moving docs
    cilk_spawn compute_gains(P.V1, P.n1, before, left2right, gains.V1);
    compute_gains(P.V2, P.n2, before, right2left, gains.V2);
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
                before[q] = deg1[q] * logn1 - deg1[q] * log2_cmp(deg1[q] + 1)
                    + deg2[q] * logn2 - deg2[q] * log2_cmp(deg2[q] + 1);
            }
            if (deg1[q]) {
                left2right[q] = (deg1[q] - 1) * logn1
                    - (deg1[q] - 1) * log2_cmp(deg1[q]) + (deg2[q] + 1) * logn2
                    - (deg2[q] + 1) * log2_cmp(deg2[q] + 2);
            }
            if (deg2[q])
                right2left[q] = (deg1[q] + 1) * logn1
                    - (deg1[q] + 1) * log2_cmp(deg1[q] + 2)
                    + (deg2[q] - 1) * logn2 - (deg2[q] - 1) * log2_cmp(deg2[q]);
        }
    }

    // (2) compute gains from moving docs
    compute_gains(P.V1, P.n1, before, left2right, gains.V1);
    compute_gains(P.V2, P.n2, before, right2left, gains.V2);

    return gains;
}

void recursive_bisection_np(progress_bar& progress, docid_node* G,
    size_t num_queries, size_t n,uint64_t depth,uint64_t max_depth)
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

        std::vector<uint8_t> query_changed(num_queries, 0);
        {
            compute_deg(partition.V1, partition.n1, deg1, query_changed);
            compute_deg(partition.V2, partition.n2, deg2, query_changed);
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
    if (depth + 1 <= max_depth) {
        if (partition.n1 > 1)
            recursive_bisection_np(
                progress, partition.V1, num_queries, partition.n1, depth + 1,max_depth);
        if (partition.n2 > 1)
            recursive_bisection_np(
                progress, partition.V2, num_queries, partition.n2, depth + 1,max_depth);

        if (partition.n1 == 1)
            progress.done(1);
        if (partition.n2 == 1)
            progress.done(1);
    } else {
        progress.done(n);
    }
}

void recursive_bisection(progress_bar& progress, docid_node* G,
    size_t num_queries, size_t n, uint64_t depth,uint64_t max_depth)
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

        std::vector<uint8_t> query_changed(num_queries, 0);
        {
            cilk_spawn compute_deg(partition.V1, partition.n1, deg1, query_changed);
            compute_deg(partition.V2, partition.n2, deg2, query_changed);
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
                std::sort(gains.V2.begin(), gains.V2.end());
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
    if (depth + 1 <= max_depth) {
        if (depth < constants::PARALLEL_SWITCH_DEPTH) {
            if (partition.n1 > 1) {
                cilk_spawn recursive_bisection(progress, partition.V1,
                    num_queries, partition.n1, depth + 1,max_depth);
            }
            if (partition.n2 > 1) {
                recursive_bisection(progress, partition.V2, num_queries,
                    partition.n2, depth + 1,max_depth);
            }
            cilk_sync;
        } else {
            if (partition.n1 > 1) {
                recursive_bisection_np(progress, partition.V1, num_queries,
                    partition.n1, depth + 1,max_depth);
            }
            if (partition.n2 > 1) {
                recursive_bisection_np(progress, partition.V2, num_queries,
                    partition.n2, depth + 1,max_depth);
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
    auto num_lists = invidx.docids.size();
    auto bg = construct_bipartite_graph(invidx, min_list_len);

    // free up some space
    invidx.clear();

    // make things faster by precomputing some logs
    log2_precomp.resize(256);
    for(size_t i = 0; i < 256; i++) { log2_precomp[i] = log2f(i); }

    {
        auto max_depth = std::max(1.0,ceil(log2(bg.num_docs)-5));
        std::cout << "recursion depth = " << max_depth << std::endl;
        timer t("recursive_bisection");
        progress_bar bp("recursive_bisection", bg.num_docs);
        recursive_bisection(bp, bg.graph.data(), bg.num_queries, bg.num_docs, 0, max_depth);
    }
    return recreate_invidx(bg, num_lists);
}
