#!/bin/bash

set -e

echo "Building EventSender Test..."

# Create build directory
BUILD_DIR="cpp_src/build_event_sender_test"
mkdir -p "$BUILD_DIR"

# Copy CMakeLists.txt if it doesn't exist
if [ ! -f "$BUILD_DIR/CMakeLists.txt" ]; then
    cp "cpp_src/CMakeLists_event_sender_test.txt" "$BUILD_DIR/CMakeLists.txt"
fi

# Navigate to build directory and build
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake .

echo "Building..."
make -j$(nproc)

echo "EventSender test built successfully!"
echo "Run with: ./cpp_src/build_event_sender_test/test_event_sender"