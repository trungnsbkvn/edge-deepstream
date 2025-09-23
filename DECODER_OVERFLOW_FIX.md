# V4L2 Video Decoder Buffer Overflow Troubleshooting Guide

## Problem
`gstv4l2videodec gst_v4l2_video_dec_loop decoder is producing too many buffers`

This error occurs when the V4L2 video decoder produces decoded frames faster than the downstream pipeline can consume them, causing a buffer overflow.

## Root Causes

1. **Pipeline Bottleneck**: Downstream processing (YOLOv8n/ArcFace) is slower than video decoding
2. **Insufficient Buffer Management**: Queues not configured for real-time processing
3. **Memory Pressure**: High system/GPU memory usage slowing processing
4. **RTSP Stream Issues**: Network jitter causing irregular frame delivery

## Applied Fixes

### 1. Code Changes in `main.py`

#### Enabled Real-time Frame Dropping
```python
REALTIME_DROP = True  # Was False
```

#### Enhanced Decoder Configuration
- Added `max-pool-size=4` to limit decoder buffer pool
- Added `output-io-mode=2` for DMABUF import mode
- Enabled `drop-frame-interval=1` for frame dropping

#### Aggressive Queue Settings
- Post-decoder queue: `max-size-buffers=2` (was 3)
- Pre-decoder queue: `max-size-buffers=3` (was 5)
- Both queues set to `leaky=2` (drop old buffers)

### 2. Configuration Updates

#### Stream Multiplexer (`config_pipeline.toml`)
```toml
# Aggressive timeout for buffer overflow prevention  
batched-push-timeout=10000      # 10ms (was 20ms)
sync-inputs=0                   # Don't wait for slow sources
nvbuf-memory-type=0             # System memory for stability
```

#### Detection Intervals
```toml
# YOLOv8n: interval=3 (process every 3rd frame)
# ArcFace: interval=1 (process every frame for accuracy)
```

### 3. Environment Variables

Applied automatically by `run_optimized.sh`:
```bash
export GST_V4L2_USE_POOL=1
export GST_V4L2_MAX_POOL_SIZE=4
export DS_RTSP_LATENCY=50
export DS_RTSP_DROP_ON_LATENCY=1
```

## Usage Instructions

### 1. Quick Fix (Recommended)
```bash
./run_optimized.sh
```

### 2. Manual Application
```bash
python3 tools/fix_decoder_overflow.py
python3 main.py config/config_pipeline.toml
```

### 3. Debug Mode
```bash
export GST_DEBUG=3
./run_optimized.sh
```

## Monitoring and Verification

### Check if Fixed
Look for these indicators of success:
- No more "producing too many buffers" errors
- Steady FPS output (30+ fps)
- Low GPU memory usage (<80%)
- Consistent latency measurements

### Performance Monitoring
```bash
# In another terminal
python3 tools/monitor_performance.py

# Or monitor GPU directly
watch -n 1 nvidia-smi
```

### Debug Commands
```bash
# Check decoder buffer pools
GST_DEBUG=v4l2*:5 python3 main.py config/config_pipeline.toml 2>&1 | grep -i pool

# Monitor queue levels
GST_DEBUG=queue*:4 python3 main.py config/config_pipeline.toml 2>&1 | grep -i "queue level"
```

## Additional Optimizations if Issue Persists

### 1. Further Reduce Resolution
```toml
# In config_pipeline.toml
width=640    # Instead of 960
height=360   # Instead of 540
```

### 2. Increase Detection Intervals
```bash
# Skip even more frames
export DETECTION_INTERVAL=5
export RECOGNITION_INTERVAL=3
```

### 3. Reduce Number of Sources
Temporarily disable some camera sources to isolate the issue:
```toml
# Comment out some sources in config_pipeline.toml
# "source_id" = "rtsp://..."
```

### 4. Hardware-Specific Fixes

#### For Jetson Devices
```bash
export NVMM_DISABLE_ALIGNMENT_CHECK=1
export NVBUF_MEMORY_TYPE=0
```

#### For High-End GPUs
```bash
export GST_V4L2_MAX_POOL_SIZE=8  # Can handle more buffers
export CUDA_DEVICE_MAX_CONNECTIONS=4
```

## System Requirements Check

### Memory Requirements
- **System RAM**: 4GB+ available
- **GPU Memory**: 2GB+ available
- **Storage**: Fast SSD recommended for alignment/recognition caches

### Verify Resources
```bash
# Check memory usage
free -h

# Check GPU memory
nvidia-smi

# Check disk I/O (if slow, could cause backup)
iostat -x 1
```

## Alternative Decoder Options

If V4L2 decoder continues to have issues, try these alternatives:

### 1. Force NVDEC (Hardware)
```python
# In main.py, prefer nvh264dec over nvv4l2decoder
for cand in ['nvh264dec', 'nvv4l2decoder', 'avdec_h264']:
```

### 2. Software Decoder (Fallback)
```python
# Use software decoder as last resort
for cand in ['avdec_h264', 'nvv4l2decoder', 'nvh264dec']:
```

### 3. Custom Decoder Pipeline
Create a custom decoder with explicit buffer management:
```python
decoder_bin = Gst.ElementFactory.make('decodebin3')
decoder_bin.set_property('max-size-buffers', 2)
decoder_bin.set_property('low-watermark', 0.1)
decoder_bin.set_property('high-watermark', 0.8)
```

## Performance Expectations After Fixes

- **Reduced Buffer Overflows**: Should eliminate or greatly reduce errors
- **Improved FPS**: 40-60 fps (was 25-30 fps)  
- **Lower Latency**: 15-25ms end-to-end (was 40-60ms)
- **Stable Memory**: GPU memory should remain below 80%
- **Better Real-time Performance**: Less frame skipping

## Rollback Instructions

If optimizations cause other issues:

1. **Restore Original Settings**
   ```bash
   git checkout config/config_pipeline.toml
   # Edit main.py: REALTIME_DROP = False
   ```

2. **Use Conservative Config**
   ```bash
   python3 main.py config/config_pipeline_original.toml
   ```

3. **Disable Environment Variables**
   ```bash
   unset GST_V4L2_USE_POOL
   unset GST_V4L2_MAX_POOL_SIZE
   # etc.
   ```

The fixes are designed to be conservative and maintain recognition quality while solving the buffer overflow issue.