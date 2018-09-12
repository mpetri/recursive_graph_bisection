#pragma once

#include <chrono>
#include <fstream>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "rgb/binary_freq_collection.hpp"
#include "rgb/binary_collection.hpp"


using namespace std::chrono;

using postings_list = std::vector<uint32_t>;

struct inverted_index {
    size_t num_postings;
    uint32_t num_docs;
    uint32_t max_doc_id;
    std::vector<postings_list> docids;
    std::vector<postings_list> freqs;
    std::vector<uint32_t> doc_lengths;
    std::vector<uint32_t> doc_id_mapping;
    void resize(size_t new_size)
    {
        docids.resize(new_size);
        freqs.resize(new_size);
    }

    size_t size() const { return docids.size(); }

    void clear()
    {
        docids.resize(0);
        freqs.resize(0);
        doc_lengths.resize(0);
        doc_id_mapping.resize(0);
    }
};

int tsfprintff(FILE* f, const char* format, ...)
{
    static std::mutex pmutex;
    std::lock_guard<std::mutex> lock(pmutex);
    va_list args;
    va_start(args, format);
    int ret = vfprintf(f, format, args);
    va_end(args);
    fflush(f);
    return ret;
}

struct timer {
    high_resolution_clock::time_point start;
    std::string name;
    timer(const std::string& _n)
        : name(_n)
    {
        tsfprintff(stdout, "START(%s)\n", name.c_str());
        start = high_resolution_clock::now();
    }
    ~timer()
    {
        auto stop = high_resolution_clock::now();
        tsfprintff(stdout, "STOP(%s) - %.3f sec\n", name.c_str(),
            duration_cast<milliseconds>(stop - start).count() / 1000.0f);
    }
};

struct progress_bar {
    high_resolution_clock::time_point start;
    size_t total;
    size_t current;
    size_t cur_percent;
    progress_bar(std::string str, size_t t)
        : total(t)
        , current(0)
        , cur_percent(0)
    {
        std::cout << str << ":" << std::endl;
        tsfprintff(stdout, "[  0/100] |");
        for (size_t i = 0; i < 50; i++)
            tsfprintff(stdout, " ");
        tsfprintff(stdout, "|\r");
    }
    progress_bar& operator++()
    {
        static std::mutex pmutex;
        std::lock_guard<std::mutex> lock(pmutex);
        current++;
        float fcp = float(current) / float(total) * 100;
        size_t cp = fcp;
        if (cp != cur_percent) {
            cur_percent = cp;
            tsfprintff(stdout, "[%3d/100] |", (int)cur_percent);
            size_t print_percent = cur_percent / 2;
            for (size_t i = 0; i < print_percent; i++)
                tsfprintff(stdout, "=");
            tsfprintff(stdout, ">");
            for (size_t i = print_percent; i < 50; i++)
                tsfprintff(stdout, " ");
            tsfprintff(stdout, "|\r");
        }
        return *this;
    }
    void done(size_t num)
    {
        static std::mutex pmutex;
        std::lock_guard<std::mutex> lock(pmutex);
        current += num;
        float fcp = float(current) / float(total) * 100;
        size_t cp = fcp;
        if (cp != cur_percent) {
            cur_percent = cp;
            tsfprintff(stdout, "[%3d/100] |", (int)cur_percent);
            size_t print_percent = cur_percent / 2;
            for (size_t i = 0; i < print_percent; i++)
                tsfprintff(stdout, "=");
            tsfprintff(stdout, ">");
            for (size_t i = print_percent; i < 50; i++)
                tsfprintff(stdout, " ");
            tsfprintff(stdout, "|\r");
        }
    }
    ~progress_bar()
    {
        tsfprintff(stdout, "[100/100] |");
        for (size_t i = 0; i < 50; i++)
            tsfprintff(stdout, "=");
        tsfprintff(stdout, ">|\n");
    }
};

int fprintff(FILE* f, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    int ret = vfprintf(f, format, args);
    va_end(args);
    fflush(f);
    return ret;
}

void quit(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "error: ");
    vfprintf(stderr, format, args);
    va_end(args);
    if (errno != 0) {
        fprintf(stderr, ": %s\n", strerror(errno));
    } else {
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    exit(EXIT_FAILURE);
}

FILE* fopen_or_fail(std::string file_name, const char* mode)
{
    FILE* out_file = fopen(file_name.c_str(), mode);
    if (!out_file) {
        quit("opening output file %s failed", file_name.c_str());
    }
    return out_file;
}

void fclose_or_fail(FILE* f)
{
    int ret = fclose(f);
    if (ret != 0) {
        quit("closing file failed");
    }
}

uint32_t read_u32(FILE* f)
{
    uint32_t x;
    int ret = fread(&x, sizeof(uint32_t), 1, f);
    if (feof(f)) {
        return 0;
    }
    if (ret != 1) {
        quit("read u32 from file failed: %d != %d", ret, 1);
    }
    return x;
}

void read_u32s(FILE* f, void* ptr, size_t n)
{
    size_t ret = fread(ptr, sizeof(uint32_t), n, f);
    if (ret != n) {
        quit("read u32s from file failed: %d != %d", ret, n);
    }
}

std::vector<uint32_t> read_uint32_list(FILE* f)
{
    uint32_t list_len = read_u32(f);
    if (list_len == 0)
        return std::vector<uint32_t>();
    std::vector<uint32_t> list(list_len);
    read_u32s(f, list.data(), list_len);
    return list;
}

size_t write_u32(FILE* f, uint32_t x)
{
    size_t ret = fwrite(&x, sizeof(uint32_t), 1u, f);
    if (ret != 1u) {
        quit("writing byte to file: %u != %u", ret, 1u);
    }
    return sizeof(uint32_t);
}

size_t write_u32s(FILE* f, uint32_t* buf, size_t n)
{
    size_t ret = fwrite(buf, sizeof(uint32_t), n, f);
    if (ret != n) {
        quit("writing byte to file: %u != %u", ret, n);
    }
    return n * sizeof(uint32_t);
}

size_t write_uint32_list(FILE* f, std::vector<uint32_t>& list)
{
    size_t written_bytes = write_u32(f, list.size());
    written_bytes += write_u32s(f, list.data(), list.size());
    return written_bytes;
}

inverted_index read_ds2i_files(std::string ds2i_prefix)
{
    inverted_index idx;
    std::string docs_file = ds2i_prefix + ".docs";
    timer t("read input list from " + docs_file);
    auto df = fopen_or_fail(docs_file, "rb");
    size_t num_docs = 0;
    size_t num_postings = 0;
    size_t num_lists = 0;
    uint32_t max_doc_id = 0;
    {
        // (1) skip the numdocs list
        read_uint32_list(df);
        // (2) keep reading lists
        while (!feof(df)) {
            const auto& list = read_uint32_list(df);
            size_t n = list.size();
            if (n == 0) {
                break;
            }
            max_doc_id = std::max(max_doc_id, list.back());
            num_lists++;
            num_postings += n;
            idx.docids.emplace_back(std::move(list));
        }
        num_docs = max_doc_id + 1;
    }
    fclose_or_fail(df);
    std::string freqs_file = ds2i_prefix + ".freqs";
    auto ff = fopen_or_fail(freqs_file, "rb");
    {
        while (!feof(ff)) {
            const auto& list = read_uint32_list(ff);
            size_t n = list.size();
            if (n == 0) {
                break;
            }
            idx.freqs.emplace_back(std::move(list));
        }
    }
    fclose_or_fail(ff);
    idx.num_docs = num_docs;
    idx.max_doc_id = max_doc_id;
    idx.num_postings = num_postings;
    std::cout << "\tnum_docs = " << num_docs << std::endl;
    std::cout << "\tmax_doc_id = " << max_doc_id << std::endl;
    std::cout << "\tnum_lists = " << num_lists << std::endl;
    std::cout << "\tnum_postings = " << num_postings << std::endl;
    return idx;
}

void write_ds2i_files(inverted_index& idx, std::string ds2i_out_prefix)
{
    std::string docs_file = ds2i_out_prefix + ".docs";
    std::string freqs_file = ds2i_out_prefix + ".freqs";
    std::string lens_file = ds2i_out_prefix + ".sizes";
    std::string mapping_file = ds2i_out_prefix + ".mapping";
    {
        auto df = fopen_or_fail(docs_file, "wb");
        {
            // ds2i: 1st list contains num docs
            std::vector<uint32_t> tmp(1);
            tmp[0] = idx.num_docs;
            write_uint32_list(df, tmp);
        }
        for (size_t i = 0; i < idx.docids.size(); i++) {
            write_uint32_list(df, idx.docids[i]);
        }
        fclose_or_fail(df);
    }
    {
        auto ff = fopen_or_fail(freqs_file, "wb");
        for (size_t i = 0; i < idx.freqs.size(); i++) {
            write_uint32_list(ff, idx.freqs[i]);
        }
        fclose_or_fail(ff);
    }
    {
        auto sf = fopen_or_fail(lens_file, "wb");
        write_uint32_list(sf, idx.doc_lengths);
        fclose_or_fail(sf);
    }
    {
        auto mf = fopen_or_fail(mapping_file, "w");
        for (size_t i = 0; i < idx.doc_id_mapping.size(); ++i) {
            fprintff(mf, "%zu %zu\n", idx.doc_id_mapping[i], i);
        }
        fclose_or_fail(mf);
    }
}

void emit(std::ostream& os, const uint32_t* vals, size_t n)
{
    os.write(reinterpret_cast<const char*>(vals), sizeof(*vals) * n);
}
void emit(std::ostream& os, uint32_t val) { emit(os, &val, 1); }
void reorder_inverted_index(const std::string& input_basename,
    const std::string& output_basename, const std::vector<uint32_t>& mapping)
{
    std::ofstream output_mapping(output_basename + ".mapping");
    emit(output_mapping, mapping.data(), mapping.size());
    binary_collection input_sizes((input_basename + ".sizes").c_str());
    auto sizes = *input_sizes.begin();
    auto num_docs = sizes.size();
    std::vector<uint32_t> new_sizes(num_docs);
    for (size_t i = 0; i < num_docs; ++i) {
        new_sizes[mapping[i]] = sizes.begin()[i];
    }
    std::ofstream output_sizes(output_basename + ".sizes");
    emit(output_sizes, sizes.size());
    emit(output_sizes, new_sizes.data(), num_docs);
    std::ofstream output_docs(output_basename + ".docs");
    std::ofstream output_freqs(output_basename + ".freqs");
    emit(output_docs, 1);
    emit(output_docs, mapping.size());
    binary_freq_collection input(input_basename.c_str());
    std::vector<std::pair<uint32_t, uint32_t>> pl;
    for (const auto& seq : input) {
        for (size_t i = 0; i < seq.docs.size(); ++i) {
            pl.emplace_back(mapping[seq.docs.begin()[i]], seq.freqs.begin()[i]);
        }
        std::sort(pl.begin(), pl.end());
        emit(output_docs, pl.size());
        emit(output_freqs, pl.size());
        for (const auto& posting : pl) {
            emit(output_docs, posting.first);
            emit(output_freqs, posting.second);
        }
        pl.clear();
    }
}
