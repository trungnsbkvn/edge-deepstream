cmake_minimum_required(VERSION 3.16)
project(PerfStatsTest VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -O2
    -g
)

# Source files for the test
set(PERF_STATS_SOURCES
    ../utils/env_utils.cpp
    ../utils/perf_stats.cpp
)

# Create test executable
add_executable(test_perf_stats
    ../test_perf_stats.cpp
    ${PERF_STATS_SOURCES}
)

# Link libraries
target_link_libraries(test_perf_stats
    pthread
)

# Print configuration
message(STATUS "PerfStats Test Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")