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
export GST_DEBUG=3  # Minimal debug output / "*:1,nvinfer:0"
export GST_PLUGIN_FEATURE_RANK="nvv4l2decoder:MAX"

# RTSP optimizations
export DS_FORCE_CONVERT_RGBA=0          # Force RGBA conversion for compatibility
export DS_RTSP_LATENCY=200              # Jitter buffer latency (ms)
export DS_RTSP_TCP=1                    # Force TCP for reliability
export DS_RTSP_DROP_ON_LATENCY=1        # Drop frames when late
export DS_RTSP_RETRANS=0                # Disable retransmission
# Advanced rtspsrc knobs (uncomment to adjust)
# export DS_RTSP_DO_RTCP=0              # Disable RTCP (default 0)
# export DS_RTSP_NTP_SYNC=0             # Disable NTP sync (default 0)
# export DS_RTSP_USER_AGENT="DeepStream/1.0"
# export DS_RTSP_BUFFER_MODE=1          # 1 = LOW_LATENCY
# export DS_RTSP_TIMEOUT_US=5000000     # 5s timeout
# export DS_RTSP_RETRY=3                # Retry count
# export DS_RTSP_TCP_TIMEOUT_US=5000000
# export DS_REALTIME_DROP=1             # Global realtime drop policy

# Decoder tuning (optional overrides)
# export DS_DEC_DROP_FRAME_INTERVAL=1   # Drop N-1 frames to keep realtime
# export DS_DEC_DISABLE_DPB=1           # Disable DPB to reduce latency
# export DS_DEC_MAX_POOL_SIZE=4         # Limit decoder buffer pool
# export DS_DEC_OUTPUT_IO_MODE=2        # 2=DMABUF_IMPORT
# export DS_DEC_NUM_EXTRA_SURFACES=0    # Extra decoder surfaces (>=0 to set)
# export DS_DEC_FORCE_PROGRESSIVE=1     # Force interlace-mode=progressive
# export DS_DEC_FORCE_FORMAT=NV12       # Force output format
# export DS_DEC_CAPS_STR="video/x-raw, format=NV12, interlace-mode=progressive, pixel-aspect-ratio=1/1" # full custom caps

# Queue settings (RTSP pre/post)
# Global defaults (apply to both unless overridden):
export DS_RTSP_QUEUE_LEAKY=2
export DS_RTSP_QUEUE_MAX_BUFFERS=3
export DS_RTSP_QUEUE_MAX_BYTES=0
export DS_RTSP_QUEUE_MAX_TIME=0
export DS_RTSP_QUEUE_SILENT=true
export DS_RTSP_QUEUE_FLUSH_ON_EOS=true
# Pre-queue overrides (before depay/parse):
# export DS_RTSP_QUEUE_PRE_LEAKY=2
#export DS_RTSP_QUEUE_PRE_MAX_BUFFERS=5
# export DS_RTSP_QUEUE_PRE_MAX_BYTES=0
# export DS_RTSP_QUEUE_PRE_MAX_TIME=0
# export DS_RTSP_QUEUE_PRE_SILENT=true
# export DS_RTSP_QUEUE_PRE_FLUSH_ON_EOS=true
# Post-decode queue overrides:
# export DS_RTSP_QUEUE_POST_LEAKY=2
#export DS_RTSP_QUEUE_POST_MAX_BUFFERS=4
# export DS_RTSP_QUEUE_POST_MAX_BYTES=0
# export DS_RTSP_QUEUE_POST_MAX_TIME=0
# export DS_RTSP_QUEUE_POST_SILENT=true
# export DS_RTSP_QUEUE_POST_FLUSH_ON_EOS=true

# Core pipeline queue settings (between major pipeline stages)
# Defaults tuned for low latency; adjust if needed
# export DS_PIPE_QUEUE_LEAKY=2
# export DS_PIPE_QUEUE_MAX_BUFFERS=3
# export DS_PIPE_QUEUE_MAX_BYTES=0
# export DS_PIPE_QUEUE_MAX_TIME=0
# export DS_PIPE_QUEUE_SILENT=true

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

# Enrollment parameters
export DS_ENROLL_DUP_THRESHOLD=0.55
export DS_ENROLL_INTRA_THRESHOLD=0.40
export DS_ENROLL_BLUR_VAR_MIN=25

# Force NVMM caps for better performance
export DS_FORCE_NVMM=1

# Recognition / probe tuning (optional)
# export RECOG_THRESH=0.35           # Override recognition threshold
# export PGIE_MIN_CONF=0.6           # Detector confidence threshold
# export PGIE_DEBUG_DETS=0           # Debug detection filtering

# FAISS toggles and debug
# export DS_DISABLE_FAISS=0
# export DS_FAISS_DEBUG=0

# Runtime behavior
# export DS_QUIT_ON_EMPTY=0          # Quit on EOS when all sources removed
# export DS_RUN_DURATION_SEC=0       # Auto-quit after N seconds (0 = disabled)
# export EVENT_SENDER_DEBUG=0

export EMBEDDING_CACHE_SIZE=100

# Run the application
echo "Starting edge-deepstream with real-time optimizations..."
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 main.py "$CONFIG_FILE"