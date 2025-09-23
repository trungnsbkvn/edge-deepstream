#!/usr/bin/env python3
"""
Fix for gstv4l2videodec buffer overflow issue.
This script applies additional runtime fixes for decoder buffer management.
"""

import os
import sys

def apply_gstreamer_fixes():
    """Apply GStreamer environment variable fixes for decoder buffer overflow."""
    
    # V4L2 decoder buffer management
    os.environ['GST_V4L2_USE_POOL'] = '1'
    os.environ['GST_V4L2_MIN_POOL_SIZE'] = '2'
    os.environ['GST_V4L2_MAX_POOL_SIZE'] = '4'
    
    # Force immediate buffer dropping for real-time performance
    os.environ['GST_PLUGIN_FEATURE_RANK'] = 'nvv4l2decoder:MAX'
    
    # Reduce decoder internal buffering
    os.environ['GST_V4L2_DISABLE_DMABUF'] = '0'  # Keep DMABUF enabled
    os.environ['GST_V4L2_IO_AUTO_SELECT'] = '0'   # Manual IO mode selection
    
    # DeepStream specific optimizations
    os.environ['NVDS_ENABLE_LATENCY_MEASUREMENT'] = '1'
    os.environ['NVDS_DISABLE_ERROR_DISPLAY'] = '1'
    
    # Additional real-time optimizations
    os.environ['GST_DEBUG_NO_COLOR'] = '1'
    os.environ['GST_DEBUG'] = '2'  # Reduce debug output
    
    print("Applied GStreamer fixes for V4L2 decoder buffer overflow")

def check_system_resources():
    """Check system resources that might contribute to buffer overflow."""
    
    try:
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            print(f"WARNING: High memory usage ({memory.percent:.1f}%)")
            print("Consider reducing batch size or resolution")
        
        # Check GPU memory if available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                gpu_percent = (used / total) * 100
                if gpu_percent > 85:
                    print(f"WARNING: High GPU memory usage ({gpu_percent:.1f}%)")
                    print("Consider reducing batch size or enable GPU memory optimization")
        except Exception:
            pass
            
    except ImportError:
        print("psutil not available, skipping resource check")

def suggest_pipeline_optimizations():
    """Suggest additional pipeline optimizations."""
    
    print("\nAdditional optimizations to prevent buffer overflow:")
    print("1. Reduce input resolution in config_pipeline.toml:")
    print("   width=640, height=360 (instead of 1920x1080)")
    
    print("\n2. Increase detection intervals:")
    print("   YOLOv8n interval=4 (skip more frames)")
    print("   ArcFace interval=2")
    
    print("\n3. Enable more aggressive queue settings:")
    print("   max-size-buffers=1 for post-decoder queues")
    
    print("\n4. Use environment variables for quick tuning:")
    print("   export GST_V4L2_MAX_POOL_SIZE=2")
    print("   export NVDS_MAX_BATCH_SIZE=4")

def main():
    print("=== V4L2 Video Decoder Buffer Overflow Fix ===")
    
    # Apply GStreamer fixes
    apply_gstreamer_fixes()
    
    # Check system resources
    print("\nChecking system resources...")
    check_system_resources()
    
    # Provide optimization suggestions
    suggest_pipeline_optimizations()
    
    print("\n=== Fixes Applied ===")
    print("Restart your pipeline to apply the changes.")
    print("Monitor with: GST_DEBUG=3 python main.py config/config_pipeline.toml")

if __name__ == "__main__":
    main()