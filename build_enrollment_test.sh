#!/bin/bash

echo "Building Enrollment Integration Test..."

# Create build directory
mkdir -p cpp_src/build_enrollment_test
cd cpp_src/build_enrollment_test

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.4 \
      -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream \
      -DOPENCV_DIR=/usr \
      -DFAISS_ROOT_DIR=/usr/local \
      -DJSON_INCLUDE_DIR=/usr/include \
      .. -B .

# Build the utilities library
echo "Building edge_utils library..."
make edge_utils -j$(nproc)

if [ $? -eq 0 ]; then
    echo "edge_utils built successfully"
else
    echo "Failed to build edge_utils"
    exit 1
fi

# Compile the enrollment integration test
echo "Compiling enrollment integration test..."

g++ -std=c++17 -O2 \
    -I../include \
    -I/usr/include/opencv4 \
    -I/usr/local/include \
    -I/usr/include/nlohmann \
    -I/usr/include/gstreamer-1.0 \
    -I/usr/include/glib-2.0 \
    -I/usr/lib/aarch64-linux-gnu/glib-2.0/include \
    ../../test_enrollment_integration.cpp \
    -L. -ledge_utils \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_objdetect \
    -lfaiss \
    -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 \
    -lpthread \
    -o test_enrollment_integration

if [ $? -eq 0 ]; then
    echo "Enrollment integration test compiled successfully!"
    echo "Executable: ./cpp_src/build_enrollment_test/test_enrollment_integration"
    
    # Copy to workspace root for convenience
    cp test_enrollment_integration ../../test_enrollment_integration
    echo "Test copied to: ./test_enrollment_integration"
else
    echo "Failed to compile enrollment integration test"
    exit 1
fi