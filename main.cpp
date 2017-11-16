#include "recursive_graph_bisection.hpp"
#include "util.hpp"

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "%s <docid_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::string docid_file = argv[1];

    auto invidx = read_d2si_docs(docid_file);

    auto reordered_invidx = reorder_docids_graph_bisection(invidx);
}