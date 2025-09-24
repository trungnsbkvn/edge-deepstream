cmake_minimum_required(VERSION 3.16)
project(EnrollOpsTest VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# FAISS (optional)
find_path(FAISS_INCLUDE_DIR
    NAMES faiss/IndexFlat.h
    PATHS /usr/local/include
          /usr/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/faiss/include)

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS /usr/local/lib
          /usr/lib
          ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/faiss/lib)

# cpptoml for TOML parsing
find_path(CPPTOML_INCLUDE_DIR
    NAMES cpptoml.h
    PATHS /usr/local/include
          /usr/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cpptoml)

# nlohmann/json for JSON parsing  
find_path(JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    PATHS /usr/local/include
          /usr/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/json/single_include)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${OpenCV_INCLUDE_DIRS}
)

if(FAISS_INCLUDE_DIR)
    include_directories(${FAISS_INCLUDE_DIR})
    add_compile_definitions(HAVE_FAISS)
endif()

if(CPPTOML_INCLUDE_DIR)
    include_directories(${CPPTOML_INCLUDE_DIR})
    add_compile_definitions(HAVE_CPPTOML)
endif()

if(JSON_INCLUDE_DIR)
    include_directories(${JSON_INCLUDE_DIR})
    add_compile_definitions(HAVE_JSON)
endif()

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -O2
    -g
)

# Source files for the test
set(ENROLL_OPS_SOURCES
    ../utils/env_utils.cpp
    ../utils/config_parser.cpp
    ../utils/faiss_index.cpp
    ../utils/tensorrt_infer.cpp
    ../utils/enroll_ops.cpp
)

# Create test executable
add_executable(test_enroll_ops
    ../test_enroll_ops.cpp
    ${ENROLL_OPS_SOURCES}
)

# Link libraries
target_link_libraries(test_enroll_ops
    ${OpenCV_LIBS}
    pthread
)

if(FAISS_LIBRARY)
    target_link_libraries(test_enroll_ops ${FAISS_LIBRARY})
endif()

# Print configuration
message(STATUS "EnrollOps Test Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV found: ${OpenCV_FOUND}")
message(STATUS "  FAISS found: ${FAISS_LIBRARY}")
message(STATUS "  CPPTOML found: ${CPPTOML_INCLUDE_DIR}")
message(STATUS "  JSON found: ${JSON_INCLUDE_DIR}")