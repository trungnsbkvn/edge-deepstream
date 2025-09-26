#!/bin/bash

# Motion-Optimized DeepStream Pipeline Runner
# Optimized for stable recognition during walking and motion scenarios

echo "Starting Motion-Optimized DeepStream Pipeline..."

# Set motion-specific environment variables for optimal performance
export DS_RTSP_LATENCY=50                    # Very low latency for motion
export DS_RTSP_DROP_ON_LATENCY=1             # Drop frames on latency buildup
export DS_RTSP_BUFFER_MODE=1                 # Low latency buffer mode
export DS_RTSP_TCP=1                         # Use TCP for more reliable streaming
export DS_RTSP_QUEUE_MAX_BUFFERS=1           # Minimal buffering for motion
export DS_RTSP_QUEUE_POST_MAX_BUFFERS=1      # Minimal post-decode buffering
export DS_RTSP_QUEUE_LEAKY=2                 # Downstream leaky mode

# Decoder optimizations for motion
export DS_DEC_DROP_FRAME_INTERVAL=0          # Drop frames when decoder is behind
export DS_DEC_DISABLE_DPB=1                  # Disable decoded picture buffer
export DS_DEC_MAX_POOL_SIZE=2                # Minimal pool size for motion
export DS_DEC_OUTPUT_IO_MODE=2               # Use DMABUF for efficiency

# Pipeline queue optimizations
export DS_PIPE_QUEUE_LEAKY=2                 # Leaky downstream
export DS_PIPE_QUEUE_MAX_BUFFERS=1           # Minimal pipeline buffering

# Performance monitoring
export PERF_VERBOSE=0                        # Enable performance logging
export PERF_STATS_PATH="/dev/shm/edge-deepstream/perf_stats_motion.json"

# Recognition thresholds (more lenient for motion scenarios)
export PGIE_MIN_CONF=0.35                    # Lower confidence for motion blur

# Create performance monitoring directory
mkdir -p /dev/shm/edge-deepstream

echo "Motion optimization environment variables set:"
echo "  - RTSP latency: ${DS_RTSP_LATENCY}ms"
echo "  - Frame dropping: Enabled"
echo "  - Buffer size: Minimal (1 frame)"
echo "  - Detection confidence: ${PGIE_MIN_CONF}"
echo "  - Performance monitoring: Enabled"

# Run the pipeline with motion optimizations
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 main.py

echo "Motion-optimized pipeline completed."