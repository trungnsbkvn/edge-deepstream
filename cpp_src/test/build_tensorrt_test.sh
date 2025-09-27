#!/bin/bash

# Build TensorRT test
set -e

echo "=== Building TensorRT Test ==="

# Determine script directory (cpp_src)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Create build directory relative to cpp_src
BUILD_DIR="$SCRIPT_DIR/build_tensorrt_test"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring TensorRT test build..."
cp "$SCRIPT_DIR/CMakeLists_tensorrt_test.txt" ./CMakeLists.txt
cp -r "$SCRIPT_DIR/include" .
cp -r "$SCRIPT_DIR/utils" .
cp "$SCRIPT_DIR/test_face_engines.cpp" .
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