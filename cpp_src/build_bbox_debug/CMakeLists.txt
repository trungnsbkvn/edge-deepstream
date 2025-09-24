cmake_minimum_required(VERSION 3.16)
project(BBoxDebug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ./include
    ./utils
    ../include
    ../utils
)

# Source files
add_executable(debug_bbox_usage
    ../../debug_bbox_usage.cpp
    ../utils/enroll_ops.cpp
    ../utils/config_parser.cpp
    ../utils/env_utils.cpp
)

target_link_libraries(debug_bbox_usage
    ${OpenCV_LIBS}
    stdc++fs
)

target_include_directories(debug_bbox_usage PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ../include
    ../utils
)

target_link_directories(debug_bbox_usage PRIVATE
    /usr/local/lib
    /opt/nvidia/deepstream/deepstream/lib
    ../../third_party/faiss/lib
)

target_compile_definitions(debug_bbox_usage PRIVATE
    OPENCV_DNN_AVAILABLE=1
)
