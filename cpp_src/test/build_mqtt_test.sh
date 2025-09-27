#!/bin/bash

# Build MQTT test
set -e

echo "=== Building MQTT Test ==="

# Create build directory
BUILD_DIR="build_mqtt_test"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring MQTT test build..."
cp ../CMakeLists_mqtt_test.txt ./CMakeLists.txt
cp -r ../include .
cp -r ../utils .
cp ../test_mqtt.cpp .
cmake .

# Build
echo "Building MQTT test..."
make -j$(nproc)

echo "Build complete!"
echo ""
echo "To run the MQTT test:"
echo "  cd $BUILD_DIR"
echo "  ./test_mqtt"
echo ""
echo "Note: Make sure you have an MQTT broker running on localhost:1883"
echo "You can install mosquitto: sudo apt install mosquitto mosquitto-clients"
echo "And test with: mosquitto_pub -h localhost -t '/local/core/v2/ai/request' -m '1;25;cam001;rtsp://test/stream;1;;;;test123'"
echo ""