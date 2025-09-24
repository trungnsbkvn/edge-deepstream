#!/bin/bash

# EdgeDeepStream C++ Build Script
set -e

echo "Building EdgeDeepStream C++ Application..."

# Create build directory
BUILD_DIR="build"
cd cpp_src

if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake configuration..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building application..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable: $(pwd)/edge_deepstream"

# Copy executable to parent directory for convenience
cp edge_deepstream ../../edge_deepstream_cpp

echo "C++ executable copied to: $(pwd)/../../edge_deepstream_cpp"