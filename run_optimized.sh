#!/bin/bash

# Startup script for edge-deepstream with decoder overflow fixes
# This script applies all necessary environment variables and configurations

echo "=== Edge DeepStream Startup with Decoder Overflow Fixes ==="

# Apply decoder buffer fixes
export GST_V4L2_USE_POOL=1
export GST_V4L2_MIN_POOL_SIZE=2
export GST_V4L2_MAX_POOL_SIZE=4
export GST_V4L2_DISABLE_DMABUF=0
export GST_V4L2_IO_AUTO_SELECT=0

# DeepStream optimizations
export NVDS_ENABLE_LATENCY_MEASUREMENT=1
export NVDS_DISABLE_ERROR_DISPLAY=1

# GStreamer optimizations for real-time performance
export GST_DEBUG_NO_COLOR=1
export GST_DEBUG=1  # Minimal debug output
export GST_PLUGIN_FEATURE_RANK="nvv4l2decoder:MAX"

# RTSP optimizations
export DS_RTSP_LATENCY=50          # Very low latency
export DS_RTSP_TCP=1               # Force TCP for reliability
export DS_RTSP_DROP_ON_LATENCY=1   # Drop frames on latency
export DS_RTSP_RETRANS=0           # Disable retransmission

# Performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MODULE_LOADING=LAZY

# Memory management
export MALLOC_ARENA_MAX=2
export G_SLICE=always-malloc

echo "Environment variables applied for decoder overflow prevention"

# Check if config file exists
CONFIG_FILE="${1:-config/config_pipeline_optimized.toml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    echo "Using default: config/config_pipeline.toml"
    CONFIG_FILE="config/config_pipeline.toml"
fi

echo "Starting pipeline with config: $CONFIG_FILE"
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
echo "Press Ctrl+C to stop"
echo ""

#Enrollment parameters
export DS_ENROLL_DUP_THRESHOLD=0.55
export DS_ENROLL_INTRA_THRESHOLD=0.40
export DS_ENROLL_BLUR_VAR_MIN=25

# Force NVMM caps for better performance
export DS_FORCE_NVMM=1

export EMBEDDING_CACHE_SIZE=1000

# Run the application
echo "Starting edge-deepstream with real-time optimizations..."
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 main.py "$CONFIG_FILE"