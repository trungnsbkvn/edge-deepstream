#!/bin/bash

# Build minimal uridecodebin test
echo "Building minimal uridecodebin test..."

g++ -o test_minimal_uridecodebin test_minimal_uridecodebin.cpp \
    $(pkg-config --cflags --libs gstreamer-1.0) \
    -std=c++17

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Usage:"
    echo "  ./test_minimal_uridecodebin rtsp://admin:123456Aa@192.168.0.213:1554/Streaming/Channels/501"
    echo ""
else
    echo "❌ Build failed!"
    exit 1
fi