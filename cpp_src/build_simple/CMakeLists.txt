cmake_minimum_required(VERSION 3.16)
project(EdgeDeepStreamMinimal VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# GStreamer packages
pkg_check_modules(GST REQUIRED gstreamer-1.0)

# Compiler flags
add_compile_options(
    ${GST_CFLAGS}
    -Wall
    -Wextra
    -O2
    -g
)

# Minimal test executable
add_executable(edge_deepstream_minimal
    ../src/minimal_main.cpp
)

target_include_directories(edge_deepstream_minimal PRIVATE
    ${GST_INCLUDE_DIRS}
)

target_link_libraries(edge_deepstream_minimal
    ${GST_LIBRARIES}
    pthread
)

# Print configuration
message(STATUS "EdgeDeepStream Minimal Configuration:")
message(STATUS "  GStreamer found: ${GST_FOUND}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")