#include "recursive_graph_bisection.hpp"
#include "util.hpp"

#include <cilk/cilk_api.h>
#include <cilk/reducer_opadd.h>

double comp_sum_log_gap(const std::vector<uint32_t>& ids,const std::vector<float>& log2_precomp)
{
    double sum_log_gaps = log2f(ids[0]+1);
    for(size_t i=1;i<ids.size();i++) {
        auto gap = ids[i]-ids[i-1];
        if(gap < 256)
            sum_log_gaps += log2_precomp[gap];
        else
            sum_log_gaps += log2f(gap);
    }
    return sum_log_gaps;
}

float compute_avg_loggap(const inverted_index& idx)
{
    std::vector<float> log2_precomp(256);
    for(size_t i = 0; i < 256; i++) { log2_precomp[i] = log2f(i); }

    cilk::reducer< cilk::op_add<double> > sum_log_gaps(0.0);
    cilk::reducer< cilk::op_add<size_t> > num_gaps(0);
    cilk_for (size_t i = idx.docids.size(); i != 0; i--) {
        *sum_log_gaps += comp_sum_log_gap(idx.docids[i-1],log2_precomp);
        *num_gaps += idx.docids[i-1].size();
    }
    return double(sum_log_gaps.get_value()) / double(num_gaps.get_value());
}


int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "%s <ds2i_prefix> <ds2i_out_prefix> <min_list_len> <num threads>\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    std::string ds2i_prefix = argv[1];
    std::string ds2i_out_prefix = argv[2];
    size_t min_list_len = atoi(argv[3]);
    if (argc == 5) {
        int threads = atoi(argv[4]);
        __cilkrts_set_param("nworkers", std::to_string(threads).c_str());
    }

    auto invidx = read_ds2i_files(ds2i_prefix);

    std::cout << "BEFORE average LogGap " 
              << compute_avg_loggap(invidx) << std::endl;

    auto reordered_invidx
        = reorder_docids_graph_bisection(invidx, min_list_len);

    std::cout << "AFTER average LogGap " 
              << compute_avg_loggap(reordered_invidx) << std::endl;

    {
        timer t("write ds2i files");
        write_ds2i_files(reordered_invidx, ds2i_out_prefix);
    }

    return EXIT_SUCCESS;
}
