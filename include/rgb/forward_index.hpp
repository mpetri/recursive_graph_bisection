#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "binary_collection.hpp"
#include "varint.hpp"
#include "util.hpp"

using id_type = std::uint32_t;

class forward_index : public std::vector<std::vector<std::uint8_t>> {
   public:
    using id_type    = uint32_t;
    using entry_type = std::vector<std::uint8_t>;

    //! Initializes a new forward index with empty containers.
    forward_index(size_t document_count, size_t term_count)
        : std::vector<entry_type>(document_count),
          m_term_count(term_count),
          m_term_counts(document_count){}

    const std::size_t &term_count() const { return m_term_count; }
    const std::size_t &term_count(id_type document) const { return m_term_counts[document]; }

    static forward_index from_inverted_index(const std::string &input_basename,
                                             size_t             min_len) {
        binary_collection coll((input_basename + ".docs").c_str());

        auto firstseq = *coll.begin();
        if (firstseq.size() != 1) {
            throw std::invalid_argument("First sequence should only contain number of documents");
        }
        auto num_docs  = *firstseq.begin();
        auto num_terms = std::distance(++coll.begin(), coll.end());

        forward_index fwd(num_docs, num_terms);
        {
            progress_bar         p("Building forward index", num_terms);
            id_type              tid = 0;
            std::vector<id_type> prev(num_docs, 0u);
            for (auto it = ++coll.begin(); it != coll.end(); ++it) {
                for (const auto &d : *it) {
                    if (it->size() >= min_len) {
                        varint::encode_single(tid - prev[d], fwd[d]);
                        prev[d] = tid;
                        fwd.m_term_counts[d]++;
                    }
                }
                p.done(1);
                ++tid;
            }
        }
        return fwd;
    }

    std::vector<id_type> terms(id_type document) const {
        const entry_type &    encoded_terms = (*this)[document];
        std::vector<uint32_t> terms;
            terms.resize(encoded_terms.size() * 5);
            size_t n = 0;
            varint::decode(encoded_terms.data(), terms.data(), encoded_terms.size(), n);
            terms.resize(n);
            terms.shrink_to_fit();
        return terms;
    }

   private:
    std::size_t              m_term_count;
    std::vector<std::size_t> m_term_counts;

};
