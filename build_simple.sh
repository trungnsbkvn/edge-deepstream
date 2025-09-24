#!/bin/bash

# EdgeDeepStream C++ Simple Build Script
set -e

echo "Building EdgeDeepStream C++ Simple Application..."

# Create build directory
BUILD_DIR="build_simple"
cd cpp_src

if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake configuration..."
cp ../CMakeLists_minimal.txt ./CMakeLists.txt
cmake . -DCMAKE_BUILD_TYPE=Release

echo "Building application..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable: $(pwd)/edge_deepstream_minimal"

# Copy executable to parent directory for convenience
cp edge_deepstream_minimal ../../edge_deepstream_minimal

echo "Minimal C++ executable copied to: $(pwd)/../../edge_deepstream_minimal"