#!/bin/bash

set -e

echo "=== Building EdgeDeepStream C++ Application ==="

cd /home/m2n/edge-deepstream/cpp_src

# Clean previous build
echo "Cleaning previous build..."
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the application
echo "Building application..."
make -j$(nproc)

echo "Build completed successfully!"
echo ""
echo "Executable: $(pwd)/edge_deepstream"
echo ""
echo "To run the application:"
echo "  ./edge_deepstream /home/m2n/edge-deepstream/config/config_pipeline.toml"
echo ""
echo "To run with specific duration:"
echo "  ./edge_deepstream /home/m2n/edge-deepstream/config/config_pipeline.toml 30000"