#pragma once

#include <chrono>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace std::chrono;

using postings_list = std::vector<uint32_t>;

using inverted_index = std::vector<postings_list>;

struct timer {
    high_resolution_clock::time_point start;
    std::string name;
    timer(const std::string& _n)
        : name(_n)
    {
        std::cerr << "START(" << name << ")" << std::endl;
        start = high_resolution_clock::now();
    }
    ~timer()
    {
        auto stop = high_resolution_clock::now();
        std::cerr << "STOP(" << name << ") - "
                  << duration_cast<milliseconds>(stop - start).count() / 1000.0f
                  << " sec" << std::endl;
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
    // for (uint32_t j = 0; j < list_len; j++) {
    //     list[j]++; // ensure there are no 0s
    // }
    return list;
}

inverted_index read_d2si_docs(std::string docs_file, int min_list_len)
{
    inverted_index idx;
    timer t("read input list from " + docs_file);
    auto df = fopen_or_fail(docs_file, "rb");
    size_t num_docs = 0;
    size_t num_postings = 0;
    size_t num_lists = 0;
    {
        // (1) skip the numdocs list
        read_uint32_list(df);
        // (2) keep reading lists
        uint32_t max_doc_id = 0;
        while (!feof(df)) {
            const auto& list = read_uint32_list(df);
            size_t n = list.size();
            if (n < min_list_len)
                continue;
            max_doc_id = std::max(max_doc_id, list.back());
            num_lists++;
            num_postings += n;
            idx.emplace_back(std::move(list));
        }
        num_docs = max_doc_id - 1;
    }
    std::cout << "num_docs = " << num_docs << std::endl;
    std::cout << "num_lists = " << num_lists << std::endl;
    std::cout << "num_postings = " << num_postings << std::endl;
    return idx;
}
