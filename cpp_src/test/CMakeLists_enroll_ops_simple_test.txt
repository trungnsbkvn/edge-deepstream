cmake_minimum_required(VERSION 3.16)
project(test_enroll_ops_simple)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

# Check for cpptoml (optional)
find_path(CPPTOML_INCLUDE_DIR cpptoml.h PATHS /usr/include /usr/local/include)
if(CPPTOML_INCLUDE_DIR)
    add_definitions(-DHAVE_CPPTOML)
    message(STATUS "Found cpptoml: ${CPPTOML_INCLUDE_DIR}")
else()
    message(WARNING "cpptoml not found, TOML parsing will be disabled")
endif()

# Find FAISS
find_library(FAISS_LIB faiss PATHS /usr/local/lib /opt/faiss/lib)
if(NOT FAISS_LIB)
    message(FATAL_ERROR "FAISS library not found")
endif()

# FAISS include directories
set(FAISS_INCLUDE_DIRS /usr/local/include /opt/faiss/include)

# Add executable
add_executable(test_enroll_ops_simple 
    ../test_enroll_ops_fixed.cpp
    ../utils/enroll_ops.cpp
    ../utils/faiss_index.cpp
    ../utils/env_utils.cpp
    ../utils/config_parser.cpp
)

# Include directories
target_include_directories(test_enroll_ops_simple PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${OpenCV_INCLUDE_DIRS}
    ${FAISS_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(test_enroll_ops_simple 
    ${OpenCV_LIBS}
    ${FAISS_LIB}
    ${GSTREAMER_LIBRARIES}
    stdc++fs
)

# Compiler flags
target_compile_options(test_enroll_ops_simple PRIVATE -Wall -Wextra -O2)