#!/bin/bash

# Maximum Performance Run Script for Edge DeepStream
# Optimized for high throughput with process utilization improvements

# Set CPU performance governor for maximum performance
echo "Setting CPU performance governor..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -w "$cpu" ]; then
        echo performance | sudo tee "$cpu" > /dev/null
    fi
done

# Set process priority and CPU affinity
echo "Setting process optimizations..."

# Increase process limits
ulimit -n 8192        # File descriptors
ulimit -u 32768       # Max user processes  
ulimit -s 16384       # Stack size

# Set environment variables for maximum performance
export GST_DEBUG=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Memory optimizations
export GST_PLUGIN_SCANNER=/usr/lib/aarch64-linux-gnu/gstreamer-1.0/gst-plugin-scanner
export GST_REGISTRY=/tmp/gst_registry.bin

# DeepStream optimizations
export NVDS_ENABLE_LATENCY_MEASUREMENT=0
export NVDS_ENABLE_DEBUG_DUMP=0

# Threading optimizations
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create tmpfs for high-speed temporary storage if not exists
if [ ! -d "/dev/shm/edge-deepstream" ]; then
    mkdir -p /dev/shm/edge-deepstream/recognized
    chmod 755 /dev/shm/edge-deepstream
fi

# Run with maximum performance configuration
echo "Starting DeepStream pipeline with maximum performance settings..."
echo "Using 4 streams instead of 7 for better resource utilization"
echo "Resolution reduced to 640x360 for maximum throughput"

# Use ionice and nice for optimal I/O and CPU scheduling
exec ionice -c 1 -n 2 nice -n -10 python3 main.py \
    --config config/config_pipeline_maxperf.toml \
    --verbose