#!/bin/bash

set -e

echo "Building EnrollOps Test..."

# Create build directory
BUILD_DIR="cpp_src/build_enroll_ops_test"
mkdir -p "$BUILD_DIR"

# Copy CMakeLists.txt if it doesn't exist
if [ ! -f "$BUILD_DIR/CMakeLists.txt" ]; then
    cp "cpp_src/CMakeLists_enroll_ops_test.txt" "$BUILD_DIR/CMakeLists.txt"
fi

# Navigate to build directory and build
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake .

echo "Building..."
make -j$(nproc)

echo "EnrollOps test built successfully!"
echo "Run with: ./cpp_src/build_enroll_ops_test/test_enroll_ops"