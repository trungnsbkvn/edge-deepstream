#!/bin/bash

set -e

echo "Building PerfStats Test..."

# Create build directory
BUILD_DIR="cpp_src/build_perf_stats_test"
mkdir -p "$BUILD_DIR"

# Copy CMakeLists.txt if it doesn't exist
if [ ! -f "$BUILD_DIR/CMakeLists.txt" ]; then
    cp "cpp_src/CMakeLists_perf_stats_test.txt" "$BUILD_DIR/CMakeLists.txt"
fi

# Navigate to build directory and build
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake .

echo "Building..."
make -j$(nproc)

echo "PerfStats test built successfully!"
echo "Run with: ./cpp_src/build_perf_stats_test/test_perf_stats"