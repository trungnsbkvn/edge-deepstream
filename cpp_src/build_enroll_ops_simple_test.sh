#!/bin/bash

set -e

echo "Building EnrollOps Simple Test..."

cd /home/m2n/edge-deepstream/cpp_src

# Clean previous build
rm -rf build_enroll_ops_simple_test
mkdir -p build_enroll_ops_simple_test
cd build_enroll_ops_simple_test

# Copy CMake file
cp ../CMakeLists_enroll_ops_simple_test.txt ./CMakeLists.txt

# Configure and build
cmake .
make -j$(nproc)

echo "Build completed successfully!"
echo "Run test with: ./test_enroll_ops_simple"