#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>

#include "pstl/algorithm"
#include "pstl/execution"

#include "util.hpp"

#include <x86intrin.h>

namespace constants {
const int MAX_ITER = 20;
}

struct docid_node {
    uint64_t initial_id;
    uint32_t* terms;
    size_t num_terms;
};

std::vector<float> log2_precomp;

float log2_cmp(uint32_t idx)
{
    if (idx < 256)
        return log2_precomp[idx];
    return log2f(idx);
}

void swap_nodes(std::reference_wrapper<docid_node>* a, std::reference_wrapper<docid_node>* b, std::vector<uint32_t>& deg1,
    std::vector<uint32_t>& deg2, std::vector<uint8_t>& queries_changed)
{
    for (size_t i = 0; i < a->get().num_terms; i++) {
        auto qry = a->get().terms[i];
        deg1[qry]--;
        deg2[qry]++;
        queries_changed[qry] = 1;
    }

    for (size_t i = 0; i < b->get().num_terms; i++) {
        auto qry = b->get().terms[i];
        deg1[qry]++;
        deg2[qry]--;
        queries_changed[qry] = 1;
    }
    std::swap(*a, *b);
}

struct bipartite_graph {
    size_t num_queries;
    size_t num_docs;
    size_t num_docs_inc_empty;
    std::vector<std::reference_wrapper<docid_node>> graph;
    std::vector<docid_node> docs;
    std::vector<uint32_t> doc_contents;
};

struct partition_t {
    std::reference_wrapper<docid_node>* V1;
    std::reference_wrapper<docid_node>* V2;
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
        if (dlist.size() >= min_list_len) {
            for (size_t pos = 0; pos < dlist.size(); pos++) {
                const auto& doc_id = dlist[pos];
                if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                    bg.graph[doc_id].get().initial_id = doc_id;
                    bg.graph[doc_id].get().terms[doc_offset[doc_id]++] = termid;
                }
            }
        }
    }
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& dlist = idx.docids[termid];
        if (dlist.size() < min_list_len) {
            for (size_t pos = 0; pos < dlist.size(); pos++) {
                const auto& doc_id = dlist[pos];
                if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                    bg.graph[doc_id].get().initial_id = doc_id;
                    bg.graph[doc_id].get().terms[doc_offset[doc_id]++] = termid;
                }
            }
        }
    }
}

bipartite_graph construct_bipartite_graph(
    inverted_index& idx, size_t min_list_len, size_t workers = 1)
{
    timer t("construct_bipartite_graph");
    bipartite_graph bg;
    bg.num_queries = idx.size();
    {
        timer t("determine doc sizes");
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
                compute_doc_sizes(idx, tmp_doc_sizes[id],
                    tmp_doc_sizes_non_pruned[id], min_doc_id, max_doc_id,
                    min_list_len);
            }
        }
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
        bg.docs.resize(idx.max_doc_id + 1);
        bg.num_docs_inc_empty = idx.max_doc_id + 1;
        bg.docs[0].terms = bg.doc_contents.data();
        bg.docs[0].num_terms = doc_sizes[0];
        auto num_terms_not_pruned = doc_sizes_non_pruned[0];
        for (size_t i = 1; i < doc_sizes.size(); i++) {
            bg.docs[i].terms
                = bg.docs[i - 1].terms + num_terms_not_pruned;
            bg.docs[i].num_terms = doc_sizes[i];
            num_terms_not_pruned = doc_sizes_non_pruned[i];
        }
        bg.graph = std::vector<std::reference_wrapper<docid_node>>(bg.docs.begin(), bg.docs.end());
    }
    {
        timer t("create forward index");
        size_t doc_ids_in_slice = idx.max_doc_id / workers;
        for (size_t id = 0; id < workers; id++) {
            size_t min_doc_id = id * doc_ids_in_slice;
            size_t max_doc_id = min_doc_id + doc_ids_in_slice;
            if (id + 1 == workers) {
                max_doc_id = idx.max_doc_id + 1;
                create_graph(bg, idx, min_doc_id, max_doc_id, min_list_len);
            } else {
                create_graph(bg, idx, min_doc_id, max_doc_id, min_list_len);
            }
        }
    }

    // Set ID for empty documents.
    for (uint32_t doc_id = 0; doc_id < idx.num_docs; ++doc_id) {
        if (bg.graph[doc_id].get().initial_id != doc_id) {
            bg.graph[doc_id].get().initial_id = doc_id;
        }
    }
    size_t num_empty = 0;
    {
        // all docs with 0 size go to the back!
        auto empty_cmp = [](const auto& a, const auto& b) {
            return a.get().num_terms > b.get().num_terms;
        };
        std::sort(bg.graph.begin(), bg.graph.end(), empty_cmp);
        auto ritr = bg.graph.end() - 1;
        auto itr = bg.graph.begin();
        while (itr != ritr) {
            if (itr->get().num_terms == 0) {
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


/* random shuffle seems to do ok */
partition_t initial_partition(std::reference_wrapper<docid_node>* G, size_t n)
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
    double gain = 0;
    std::reference_wrapper<docid_node>* node = nullptr;

    move_gain() = default;
    move_gain(double g, std::reference_wrapper<docid_node>* n)
        : gain(g)
        , node(n)
    {
    }
    bool operator<(const move_gain& other) const { return gain > other.gain; }
};

struct move_gains_t {
    std::vector<move_gain> V1;
    std::vector<move_gain> V2;
};

move_gain compute_single_gain(
    std::reference_wrapper<docid_node>* doc, std::vector<float>& before, std::vector<float>& after)
{
    __m128 vsum = _mm_set1_ps(0);           // initialise vector of four partial 32 bit sums
    float gain[4];
    size_t n = doc->get().num_terms / 4;
    size_t m = doc->get().num_terms % 4;

    for (size_t j = 0; j < n; j++) {
        auto q0 = doc->get().terms[j];
        auto q1 = doc->get().terms[j+1];
        auto q2 = doc->get().terms[j+2];
        auto q3 = doc->get().terms[j+3];
        __m128 val = _mm_set_ps (before[q0]- after[q0], before[q1]- after[q1], before[q2]- after[q2], before[q3]- after[q3]);
        vsum = _mm_add_ps(vsum, val);
    }
    _mm_store_ps(gain, vsum);
    auto total = gain[0] + gain[1] + gain[2] + gain[3];

    for (size_t j = 0; j < m; j++) {
        auto q = doc->get().terms[n * 4 + j];
        total += before[q]- after[q];
    }

    return move_gain(total, doc);
}

static size_t generation = 1;

template <class T> class cache_entry {
public:
    const T& value() { return m_value; }
    bool has_value() { return m_generation == generation; }
    void operator=(const T& v)
    {
        m_value = v;
        m_generation = generation;
    }

private:
    T m_value;
    size_t m_generation = 0;
};

using cache = std::vector<cache_entry<double>>;

void compute_deg(std::reference_wrapper<docid_node>* docs, size_t n, std::vector<uint32_t>& deg, std::vector<uint8_t>& query_changed)
{
    for (size_t i = 0; i < n; i++) {
        auto doc = docs + i;
        for (size_t j = 0; j < doc->get().num_terms; j++) {
            deg[doc->get().terms[j]]++;
            query_changed[doc->get().terms[j]] = 1;
        }
    }
}

template <bool isParallel = true>
void compute_gains(std::reference_wrapper<docid_node>* docs, size_t n, std::vector<float>& before,
    std::vector<float>& after, std::vector<move_gain>& res)
{
    res.resize(n);
    auto body = [&](auto&& i) {
        auto doc = docs + i;
        res[i] = compute_single_gain(doc, before, after);
    };
    if constexpr (isParallel) {
        tbb::parallel_for(size_t(0), n, [&](size_t i) { body(i); });
    } else {
        for (size_t i = 0; i < n; i++) {
            body(i);
        }
    }
}

template <bool isParallel = true>
move_gains_t compute_move_gains(partition_t& P, size_t num_queries,
    std::vector<uint32_t>& deg1, std::vector<uint32_t>& deg2,
    std::vector<float>& before, std::vector<float>& left2right,
    std::vector<float>& right2left, std::vector<uint8_t>& qry_changed)
{
    move_gains_t gains;

    float logn1 = log2f(P.n1);
    float logn2 = log2f(P.n2);

    auto body = [&](auto&& q) {
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
    };
    if constexpr (isParallel) {
        tbb::parallel_for(size_t(0), num_queries, [&](size_t q) { body(q); });
    } else {
        for (size_t q = 0; q < num_queries; q++) {
            body(q);
        }
    }

    // (2) compute gains from moving docs
    if constexpr (isParallel) {
        tbb::parallel_invoke(
            [&] { compute_gains(P.V1, P.n1, before, left2right, gains.V1); },
            [&] { compute_gains(P.V2, P.n2, before, right2left, gains.V2); });
    } else {
        compute_gains<false>(P.V1, P.n1, before, left2right, gains.V1);
        compute_gains<false>(P.V2, P.n2, before, right2left, gains.V2);
    }
    return gains;
}

template <bool isParallel = true>
void recursive_bisection(progress_bar& progress, std::reference_wrapper<docid_node>* G,
    size_t num_queries, size_t n, uint64_t depth, uint64_t max_depth,
    size_t parallel_switch_depth)
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
            if constexpr (isParallel) {
                tbb::parallel_invoke(
                    [&] { compute_deg(partition.V1, partition.n1, deg1, query_changed); },
                    [&] { compute_deg(partition.V2, partition.n2, deg2, query_changed); });
            } else {
                compute_deg(partition.V1, partition.n1, deg1, query_changed);
                compute_deg(partition.V2, partition.n2, deg2, query_changed);
            }
        }

        // (3) perform bisection. constant number of iterations
        for (int cur_iter = 1; cur_iter <= constants::MAX_ITER; cur_iter++) {
            // (3a) compute move gains
            move_gains_t gains;
            if constexpr (isParallel) {
                gains = compute_move_gains(partition, num_queries, deg1, deg2,
                    before, left2right, right2left, query_changed);
            } else {
                gains = compute_move_gains(partition, num_queries, deg1, deg2,
                    before, left2right, right2left, query_changed);
            }
            memset(query_changed.data(), 0, num_queries);

            // (3b) sort by decreasing gain. O(n log n)
            {
                if constexpr (isParallel) {
                    tbb::parallel_invoke(
                        [&] {
                            std::sort(std::execution::par_unseq,
                                gains.V1.begin(), gains.V1.end());
                        },
                        [&] {
                            std::sort(std::execution::par_unseq,
                                gains.V2.begin(), gains.V2.end());
                        });
                } else {
                    std::sort(std::execution::unseq, gains.V1.begin(),
                        gains.V1.end());
                    std::sort(std::execution::unseq, gains.V2.begin(),
                        gains.V2.end());
                }
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
        if (depth < parallel_switch_depth) {
            tbb::parallel_invoke(
                [&] {
                    recursive_bisection(progress, partition.V1, num_queries,
                        partition.n1, depth + 1, max_depth,
                        parallel_switch_depth);
                },
                [&] {
                    recursive_bisection(progress, partition.V2, num_queries,
                        partition.n2, depth + 1, max_depth,
                        parallel_switch_depth);
                });
        } else {
            if (partition.n1 > 1) {
                recursive_bisection<false>(progress, partition.V1, num_queries,
                    partition.n1, depth + 1, max_depth, parallel_switch_depth);
            }
            if (partition.n2 > 1) {
                recursive_bisection<false>(progress, partition.V2, num_queries,
                    partition.n2, depth + 1, max_depth, parallel_switch_depth);
            }
        }
        if (partition.n1 == 1)
            progress.done(1);
        if (partition.n2 == 1)
            progress.done(1);
    } else {
        auto by_id = [](auto &&lhs, auto &&rhs) {
            return lhs.get().initial_id < rhs.get().initial_id;
        };
        std::sort(
            std::execution::unseq, G, G + n, by_id);
        progress.done(n);
    }
}

auto get_mapping = [](const auto &bg) {
    std::vector<uint32_t> mapping(bg.num_docs_inc_empty, 0u);
    size_t                p = 0;
    for (const auto &g : bg.graph) {
        mapping[g.get().initial_id] = p++;
    }
    return mapping;
};

std::vector<uint32_t> reorder_docids_graph_bisection(
    inverted_index& invidx, size_t min_list_len, size_t parallel_switch_depth)
{
    auto bg = construct_bipartite_graph(invidx, min_list_len);

    // free up some space
    invidx.clear();

    // make things faster by precomputing some logs
    log2_precomp.resize(256);
    for (size_t i = 0; i < 256; i++) {
        log2_precomp[i] = log2f(i);
    }

    {
        auto max_depth = std::max(1.0, ceil(log2(bg.num_docs) - 5));
        std::cout << "recursion depth = " << max_depth << std::endl;
        timer t("recursive_bisection");
        progress_bar bp("recursive_bisection", bg.num_docs);
        recursive_bisection(bp, bg.graph.data(), bg.num_queries, bg.num_docs, 0,
            max_depth, parallel_switch_depth);
    }
    return get_mapping(bg);
}
