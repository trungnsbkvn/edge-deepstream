#!/bin/bash

# Build TensorRT test
set -e

echo "=== Building TensorRT Test ==="

# Create build directory
BUILD_DIR="build_tensorrt_test"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring TensorRT test build..."
cp ../CMakeLists_tensorrt_test.txt ./CMakeLists.txt
cp -r ../include .
cp -r ../utils .
cp ../test_face_engines.cpp .
cmake .

# Build
echo "Building TensorRT test..."
make -j$(nproc)

echo "Build complete!"
echo ""
echo "To run the TensorRT test:"
echo "  cd $BUILD_DIR"
echo "  ./test_tensorrt"
echo ""