#include "recursive_graph_bisection.hpp"
#include "util.hpp"

#include <cilk/cilk_api.h>

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

    auto reordered_invidx
        = reorder_docids_graph_bisection(invidx, min_list_len);

    {
        timer t("write ds2i files");
        write_ds2i_files(reordered_invidx, ds2i_out_prefix);
    }

    return EXIT_SUCCESS;
}