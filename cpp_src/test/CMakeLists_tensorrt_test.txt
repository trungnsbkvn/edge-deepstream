cmake_minimum_required(VERSION 3.16)
project(TensorRTTest VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

# Optional: TensorRT (for real implementation)
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu 
          /usr/local/include 
          /opt/tensorrt/include
          /usr/include)

find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu 
          /usr/local/lib 
          /opt/tensorrt/lib
          /usr/lib)

# CUDA (required for TensorRT)
find_package(CUDA QUIET)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Check for TensorRT availability
if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY AND CUDA_FOUND)
    message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
    include_directories(${TENSORRT_INCLUDE_DIR})
    include_directories(${CUDA_INCLUDE_DIRS})
    add_compile_definitions(HAVE_TENSORRT)
    set(TENSORRT_LIBS ${TENSORRT_NVINFER_LIBRARY} ${CUDA_LIBRARIES})
else()
    message(STATUS "TensorRT not found - using placeholder implementation")
    set(TENSORRT_LIBS "")
endif()

# Source files
set(SOURCES
    test_face_engines.cpp
    utils/tensorrt_infer.cpp
)

# Create test executable
add_executable(test_tensorrt ${SOURCES})

# Link libraries
target_link_libraries(test_tensorrt
    ${OpenCV_LIBS}
    ${TENSORRT_LIBS}
)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(test_tensorrt PRIVATE -Wall -Wextra -O2)
endif()

message(STATUS "TensorRT test configuration:")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
if(TENSORRT_INCLUDE_DIR)
    message(STATUS "  TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  TensorRT library: ${TENSORRT_NVINFER_LIBRARY}")
endif()
if(CUDA_FOUND)
    message(STATUS "  CUDA version: ${CUDA_VERSION}")
endif()