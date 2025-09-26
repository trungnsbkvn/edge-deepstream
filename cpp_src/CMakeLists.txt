cmake_minimum_required(VERSION 3.16)
project(EdgeDeepStream VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# GStreamer packages
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)

# DeepStream SDK
find_path(DEEPSTREAM_INCLUDE_DIR
    NAMES nvds_analytics_meta.h
    PATHS /opt/nvidia/deepstream/deepstream/sources/includes
          /opt/nvidia/deepstream/deepstream-*/sources/includes
    REQUIRED)

find_library(DEEPSTREAM_NVDS_META
    NAMES nvds_meta
    PATHS /opt/nvidia/deepstream/deepstream/lib
          /opt/nvidia/deepstream/deepstream-*/lib
    REQUIRED)

find_library(DEEPSTREAM_NVBUFSURFACE
    NAMES nvbufsurface
    PATHS /opt/nvidia/deepstream/deepstream/lib
          /opt/nvidia/deepstream/deepstream-*/lib
    REQUIRED)

find_library(DEEPSTREAM_NVDSGST_META
    NAMES nvdsgst_meta
    PATHS /opt/nvidia/deepstream/deepstream/lib
          /opt/nvidia/deepstream/deepstream-*/lib
    REQUIRED)

# CUDA
find_package(CUDA REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# FAISS (optional)
find_path(FAISS_INCLUDE_DIR
    NAMES faiss/IndexFlat.h
    PATHS /usr/local/include
          /usr/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/faiss/include)

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS /usr/local/lib
          /usr/lib
          ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/faiss/lib)

# toml11 for TOML parsing
find_path(TOML11_INCLUDE_DIR
    NAMES toml.hpp
    PATHS /usr/local/include
          /usr/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/toml11/single_include)

# nlohmann/json for JSON parsing
find_path(JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    PATHS /usr/local/include
          /usr/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/single_include)

# paho-mqtt for MQTT
find_library(PAHO_MQTT_CPP_LIBRARY
    NAMES paho-mqttpp3
    PATHS /usr/local/lib /usr/lib)

find_library(PAHO_MQTT_C_LIBRARY
    NAMES paho-mqtt3as
    PATHS /usr/local/lib /usr/lib)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_roi_helper
    ${GST_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

if(FAISS_INCLUDE_DIR)
    include_directories(${FAISS_INCLUDE_DIR})
    add_compile_definitions(HAVE_FAISS)
endif()

if(TOML11_INCLUDE_DIR)
    include_directories(${TOML11_INCLUDE_DIR})
    add_compile_definitions(HAVE_TOML11)
endif()

if(JSON_INCLUDE_DIR)
    include_directories(${JSON_INCLUDE_DIR})
    add_compile_definitions(HAVE_JSON)
endif()

# Compiler flags
add_compile_options(
    ${GST_CFLAGS}
    -Wall
    -Wextra
    -O2
    -g
)

# Source files
set(UTIL_SOURCES
    utils/env_utils.cpp
    utils/config_parser.cpp
    utils/status_codes.cpp
    utils/bus_call.cpp
    utils/event_sender.cpp
    utils/mqtt_listener.cpp
    utils/faiss_index.cpp
    utils/gen_feature.cpp
    utils/enroll_ops.cpp
    utils/perf_stats.cpp
    utils/probe.cpp
)

set(MAIN_SOURCES
    src/main.cpp
    src/pipeline.cpp
    src/source_bin.cpp
    src/probe.cpp
)

# Create utility library
add_library(edge_utils STATIC ${UTIL_SOURCES})
target_link_libraries(edge_utils
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

if(FAISS_LIBRARY)
    target_link_libraries(edge_utils ${FAISS_LIBRARY})
endif()

if(PAHO_MQTT_CPP_LIBRARY AND PAHO_MQTT_C_LIBRARY)
    target_link_libraries(edge_utils ${PAHO_MQTT_CPP_LIBRARY} ${PAHO_MQTT_C_LIBRARY})
    add_compile_definitions(HAVE_MQTT)
endif()

# ROI helper library (if exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/roi_helper)
    add_subdirectory(cpp/roi_helper)
    target_link_libraries(edge_utils roi_helper)
endif()

# Main executable
add_executable(edge_deepstream ${MAIN_SOURCES})
target_link_libraries(edge_deepstream
    edge_utils
    ${GST_LIBRARIES}
    ${GST_APP_LIBRARIES}
    ${DEEPSTREAM_NVDS_META}
    ${DEEPSTREAM_NVBUFSURFACE}
    ${DEEPSTREAM_NVDSGST_META}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    pthread
)

# YOLOv8n test executable
add_executable(test_yolov8n_only test_yolov8n_only.cpp)
target_include_directories(test_yolov8n_only PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GST_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_yolov8n_only
    ${GST_LIBRARIES}
    ${GST_APP_LIBRARIES}
    ${DEEPSTREAM_NVDS_META}
    ${DEEPSTREAM_NVBUFSURFACE}
    ${DEEPSTREAM_NVDSGST_META}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    pthread
)

# Install targets
install(TARGETS edge_deepstream DESTINATION bin)
install(TARGETS edge_utils DESTINATION lib)

# Copy config files
install(DIRECTORY config/ DESTINATION share/edge_deepstream/config)

# Print configuration summary
message(STATUS "EdgeDeepStream Configuration:")
message(STATUS "  GStreamer found: ${GST_FOUND}")
message(STATUS "  DeepStream includes: ${DEEPSTREAM_INCLUDE_DIR}")
message(STATUS "  CUDA found: ${CUDA_FOUND}")
message(STATUS "  OpenCV found: ${OpenCV_FOUND}")
message(STATUS "  FAISS found: ${FAISS_LIBRARY}")
message(STATUS "  MQTT found: ${PAHO_MQTT_CPP_LIBRARY}")
message(STATUS "  TOML11 found: ${TOML11_INCLUDE_DIR}")
message(STATUS "  JSON found: ${JSON_INCLUDE_DIR}")