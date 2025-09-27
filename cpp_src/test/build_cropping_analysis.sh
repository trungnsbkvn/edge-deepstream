#!/bin/bash

echo "Building cropping analysis test..."

# Create build directory
mkdir -p cpp_src/build_cropping_analysis
cd cpp_src/build_cropping_analysis

# Create CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(CroppingAnalysisTest)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(../include)

# Source files - minimal set for this test
set(SOURCES
    ../utils/enroll_ops.cpp
    ../utils/config_parser.cpp
    ../utils/env_utils.cpp
)

# Add executable
add_executable(test_cropping_analysis
    ../../test_cropping_analysis.cpp
    ${SOURCES}
)

# Link libraries
target_link_libraries(test_cropping_analysis
    ${OpenCV_LIBS}
    pthread
)
EOF

# Build
echo "Configuring with CMake..."
cmake .

echo "Building..."
make

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "Running cropping analysis test..."
    cd ../..
    ./cpp_src/build_cropping_analysis/test_cropping_analysis
else
    echo "❌ Build failed!"
    exit 1
fi