#!/bin/bash

# EdgeDeepStream FAISS Test Build Script
set -e

echo "Building EdgeDeepStream FAISS Test..."

# Create build directory
BUILD_DIR="build_faiss_test"
cd cpp_src

if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake configuration for FAISS test..."
cp ../CMakeLists_faiss_test.txt ./CMakeLists.txt
cmake . -DCMAKE_BUILD_TYPE=Release

echo "Building FAISS test..."
make -j$(nproc)

echo "FAISS test build completed successfully!"
echo "Executable: $(pwd)/test_faiss"

# Copy executable to parent directory for convenience
cp test_faiss ../../test_faiss_cpp

cd ../..
echo "Running FAISS test..."
./test_faiss_cpp