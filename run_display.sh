#!/bin/bash

# Script to run EdgeDeepStream pipeline with display
export DISPLAY=:1

echo "Starting EdgeDeepStream pipeline with OSD display..."
echo "Press Ctrl+C to stop"

cd /home/m2n/edge-deepstream/cpp_src/build
./test_yolov8n_only