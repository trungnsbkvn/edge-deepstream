cmake_minimum_required(VERSION 3.16)
project(EdgeDeepStreamFAISSTest VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# FAISS library
find_path(FAISS_INCLUDE_DIR
    NAMES faiss/IndexFlat.h
    PATHS /usr/local/include 
          /usr/include
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/include
    REQUIRED)

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS /usr/local/lib 
          /usr/lib
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/lib
    REQUIRED)

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -O2
    -g
)

# Include directories
include_directories(
    ${FAISS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Add FAISS definition
add_compile_definitions(HAVE_FAISS)

# FAISS test executable
add_executable(test_faiss
    ../test_faiss.cpp
    ../utils/faiss_index.cpp
)

target_include_directories(test_faiss PRIVATE
    ${FAISS_INCLUDE_DIR}
    ../include
)

target_link_libraries(test_faiss 
    ${FAISS_LIBRARY}
    pthread
)

message(STATUS "FAISS Test Configuration:")
message(STATUS "  FAISS include: ${FAISS_INCLUDE_DIR}")
message(STATUS "  FAISS library: ${FAISS_LIBRARY}")