#!/bin/bash

echo "Building crop analysis test..."

# Create build directory
mkdir -p cpp_src/build_crop_analysis
cd cpp_src/build_crop_analysis

# Create CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(CropAnalysisTest)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(../include)
include_directories(/usr/local/cuda/include)
include_directories(/opt/nvidia/deepstream/deepstream/sources/includes)
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${GLIB_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    ../utils/enroll_ops.cpp
    ../utils/faiss_index.cpp
    ../utils/config_parser.cpp
    ../utils/env_utils.cpp
    ../utils/tensorrt_infer.cpp
    ../utils/mqtt_listener.cpp
    ../utils/event_sender.cpp
    ../utils/perf_stats.cpp
    ../utils/probe.cpp
)

# Add executable
add_executable(test_crop_analysis
    ../../test_crop_analysis.cpp
    ${SOURCES}
)

# Link libraries
target_link_libraries(test_crop_analysis
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GLIB_LIBRARIES}
    faiss
    cudart
    cublas
    curand
    cusolver
    cudnn
    nvinfer
    nvinfer_plugin
    pthread
)

target_link_directories(test_crop_analysis PRIVATE
    /usr/local/cuda/lib64
    /opt/nvidia/deepstream/deepstream/lib
)
EOF

# Build
echo "Configuring with CMake..."
cmake .

echo "Building..."
make

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "Running crop analysis test..."
    cd ../..
    ./cpp_src/build_crop_analysis/test_crop_analysis
else
    echo "❌ Build failed!"
    exit 1
fi