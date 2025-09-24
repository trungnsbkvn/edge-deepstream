cmake_minimum_required(VERSION 3.10)
project(RealFaceEnrollmentTest)

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
add_executable(test_real_face_enrollment
    ../../test_real_face_enrollment.cpp
    ${SOURCES}
)

# Link libraries
target_link_libraries(test_real_face_enrollment
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

target_link_directories(test_real_face_enrollment PRIVATE
    /usr/local/cuda/lib64
    /opt/nvidia/deepstream/deepstream/lib
)
