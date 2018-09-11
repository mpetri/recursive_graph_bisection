add_executable(recursive_graph_bisection rgb.cpp)
target_link_libraries(recursive_graph_bisection
  rgb
  CLI11
  ParallelSTL
)
