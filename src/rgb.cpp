#include <thread>

#include "rgb/rgb.hpp"
#include "rgb/util.hpp"
#include "rgb/forward_index.hpp"

#include "CLI/CLI.hpp"
#include "tbb/task_scheduler_init.h"


double comp_sum_log_gap(
    const std::vector<uint32_t>& ids, const std::vector<float>& log2_precomp)
{
    double sum_log_gaps = log2f(ids[0] + 1);
    for (size_t i = 1; i < ids.size(); i++) {
        auto gap = ids[i] - ids[i - 1];
        if (gap < 256)
            sum_log_gaps += log2_precomp[gap];
        else
            sum_log_gaps += log2f(gap);
    }
    return sum_log_gaps;
}

float compute_avg_loggap(const inverted_index& idx)
{
    std::vector<float> log2_precomp(256);
    for (size_t i = 0; i < 256; i++) {
        log2_precomp[i] = log2f(i);
    }

    double sum_log_gaps(0.0);
    size_t num_gaps(0);
    for (size_t i = idx.docids.size(); i != 0; i--) {
        sum_log_gaps += comp_sum_log_gap(idx.docids[i - 1], log2_precomp);
        num_gaps += idx.docids[i - 1].size();
    }
    return double(sum_log_gaps) / double(num_gaps);
}

int main(int argc, char** argv)
{
    std::string input_basename;
    std::string output_basename;
    size_t min_len = 0;
    size_t threads = std::thread::hardware_concurrency();

    CLI::App app{ "Recursive graph bisection algorithm used for inverted "
                  "indexed reordering." };
    app.add_option("-c,--collection", input_basename, "Collection basename")
        ->required();
    app.add_option("-o,--output", output_basename, "Output basename")
        ->required();
    app.add_option("-m,--min-len", min_len, "Minimum list threshold");
    app.add_option("-t,--threads", threads, "Thread count");
    CLI11_PARSE(app, argc, argv);

    tbb::task_scheduler_init init(threads);

    auto fwd = forward_index::from_inverted_index(input_basename, min_len);

    // std::cout << "BEFORE average LogGap " << compute_avg_loggap(invidx)
    //           << std::endl;

    auto parallel_switch_depth = std::log2(threads);

    auto mapping = reorder_docids_graph_bisection(fwd, min_len, parallel_switch_depth);
    fwd.clear();
    // std::cout << "AFTER average LogGap " << compute_avg_loggap(reordered_invidx)
    //           << std::endl;

    {
        timer t("write reordered inverted index");
        reorder_inverted_index(input_basename, output_basename, mapping);
    }

    return EXIT_SUCCESS;
}
