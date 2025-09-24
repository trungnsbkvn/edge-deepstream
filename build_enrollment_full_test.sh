#!/bin/bash

echo "Building full enrollment system test..."

g++ -std=c++17 -O2 \
    -I/usr/local/cuda/include \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/include/gstreamer-1.0 \
    -I/usr/include/glib-2.0 \
    -I/usr/lib/x86_64-linux-gnu/glib-2.0/include \
    -I/usr/include/opencv4 \
    test_enrollment_full.cpp \
    cpp_src/utils/enroll_ops.cpp \
    cpp_src/utils/faiss_index.cpp \
    cpp_src/utils/config_parser.cpp \
    cpp_src/utils/env_utils.cpp \
    cpp_src/utils/tensorrt_infer.cpp \
    -L/usr/local/cuda/lib64 \
    -L/opt/nvidia/deepstream/deepstream/lib \
    -L/usr/lib/x86_64-linux-gnu \
    -lfaiss \
    -lcudart \
    -lcublas \
    -lcurand \
    -lcusolver \
    -lcudnn \
    -lnvinfer \
    -lnvinfer_plugin \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_highgui \
    -lopencv_dnn \
    -ltoml11 \
    -ljsoncpp \
    -o test_enrollment_full

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Running full enrollment test..."
    ./test_enrollment_full
else
    echo "Build failed!"
    exit 1
fi