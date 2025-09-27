#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_engine_test"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cat > CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(EngineSmokeTest LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include /opt/nvidia/tensorrt/include /opt/tensorrt/include)
find_library(TENSORRT_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib /opt/nvidia/tensorrt/lib /opt/tensorrt/lib)
find_library(CUDART_LIB cudart
    PATHS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    PATHS /usr/local/cuda/include /usr/include)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB OR NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIB)
    message(FATAL_ERROR "Required dependencies not found. Need TensorRT (headers+lib) and CUDA (headers + cudart).")
endif()

include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})

add_executable(test_engine_load ../test_engine_load.cpp)
target_link_libraries(test_engine_load ${TENSORRT_LIB} ${CUDART_LIB})
EOF

cmake .
make -j$(nproc)

echo "Built test_engine_load at $BUILD_DIR/test_engine_load"
