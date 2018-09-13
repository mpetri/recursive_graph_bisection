EXECUTE_PROCESS(COMMAND git submodule update --init
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
                OUTPUT_QUIET
)


# Add CLI11
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/CLI11 EXCLUDE_FROM_ALL)

# Add TBB
set(TBB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/tbb/)
include(${TBB_ROOT}/cmake/TBBBuild.cmake)
tbb_build(
    TBB_ROOT ${TBB_ROOT}
    CONFIG_DIR TBB_DIR)

# Add ParallelSTL
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parallelstl EXCLUDE_FROM_ALL)
