#!/bin/bash

# Environment variables for real-time RTSP streaming
export DS_RTSP_LATENCY=100          # Reduce from 150ms to 100ms
export DS_RTSP_DROP_ON_LATENCY=1    # Already set, but ensure it's active
export DS_RTSP_TCP=1                # Use TCP for reliability
export DS_RTSP_RETRANS=0           # Disable retransmission for lower latency
#export DS_CONFIG_PATH=config/config_pipeline_realtime.toml
# Enable realtime frame dropping globally
export DS_REALTIME_DROP=1

# Force NVMM caps for better performance
export DS_FORCE_NVMM=1

# Reduce buffer sizes for minimal latency
export GST_DEBUG=3                  # Enable debugging to monitor drops

echo "Starting edge-deepstream with real-time optimizations..."
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 main.py