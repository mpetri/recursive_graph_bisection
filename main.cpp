#include "recursive_graph_bisection.hpp"
#include "util.hpp"

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "%s <docid_file> <min_list_len>\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::string docid_file = argv[1];
    size_t min_list_len = atoi(argv[2]);

    auto invidx = read_d2si_docs(docid_file, min_list_len);

    auto reordered_invidx = reorder_docids_graph_bisection(invidx);
}