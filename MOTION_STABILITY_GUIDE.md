# Motion Stability Optimization Guide

## Problem Analysis
The pipeline experienced image drift and fragmentation during motion due to:

1. **Processing Bottleneck**: SGIE processing (3-6ms) much slower than PGIE (0.4-0.7ms)
2. **Low Throughput**: Only processing ~2-3 FPS instead of real-time 25-30 FPS
3. **Buffer Accumulation**: Frames queuing faster than processing during motion
4. **Tracker Lag**: Motion prediction insufficient for walking scenarios
5. **Synchronization Issues**: Different processing speeds causing frame misalignment

## Optimizations Implemented

### 1. Resolution Optimization
- **Before**: 1280x720
- **After**: 960x540
- **Impact**: ~40% reduction in processing load

### 2. Frame Processing Intervals
- **Before**: Every frame (interval=0)
- **After**: Every 2nd frame (interval=1)
- **Impact**: 50% reduction in detection load

### 3. Detection Thresholds (Motion-Friendly)
- **Before**: pre-cluster-threshold=0.50
- **After**: pre-cluster-threshold=0.35
- **Impact**: Better detection of faces with motion blur

### 4. Tracker Motion Prediction
- **StateEstimator**: Upgraded from SIMPLE(1) to REGULAR(2)
- **processNoiseVar4Loc**: Increased from 2.0 to 4.0 (higher motion variance)
- **processNoiseVar4Vel**: Increased from 0.1 to 0.5 (better velocity tracking)
- **filterLr**: Increased from 0.075 to 0.1 (faster adaptation)

### 5. Buffer Management
- **RTSP Queue**: Reduced from 3 to 1-2 buffers
- **Post-Decode Queue**: Reduced to 1 buffer
- **Batched Push Timeout**: Reduced from 25ms to 15ms

### 6. Recognition Optimization
- **recognize_once_per_track**: Disabled (0) for motion scenarios
- **max_track_embeddings**: Reduced from 60 to 30
- **min_embeddings_for_fusion**: Reduced from 2 to 1
- **fusion_mode**: Changed to "median" (more robust to outliers)

### 7. Real-time Frame Dropping
- **drop-frame-interval**: Set to 0 (adaptive dropping)
- **disable-dpb**: Enabled for faster decoding
- **max-pool-size**: Reduced to 2-4 buffers

## Usage

### Quick Start (Motion Scenarios)
```bash
./run_motion_optimized.sh
```

### Manual Environment Setup
```bash
# For testing specific motion scenarios
export DS_RTSP_LATENCY=50
export DS_RTSP_DROP_ON_LATENCY=1
export DS_RTSP_QUEUE_MAX_BUFFERS=1
export PGIE_MIN_CONF=0.35
python3 main.py
```

## Expected Performance Improvements

### Frame Rate
- **Before**: 2-3 FPS
- **After**: 8-15 FPS (target 25+ FPS)

### Motion Handling
- **Walking**: Stable tracking and recognition
- **Sudden Movement**: Better motion prediction
- **Blurred Faces**: Lower thresholds accommodate motion blur

### Latency
- **End-to-End**: Reduced from ~300ms to <100ms
- **Buffer Accumulation**: Prevented through aggressive dropping

## Monitoring

Use the performance monitoring to verify improvements:
```bash
tail -f /dev/shm/edge-deepstream/perf_stats_motion.json
```

Key metrics to watch:
- `pgie_fps` should increase to 8-15+
- `sgie_ms` should decrease below 3ms
- `pipeline_stats.detections_per_second` should increase

## Troubleshooting

### If FPS Still Low
1. Further reduce resolution in config_pipeline.toml
2. Increase detection interval in config_yolov8n_face.txt
3. Enable GPU acceleration (`use_gpu=1` in recognition section)

### If Missing Detections
1. Lower `pre-cluster-threshold` in config_yolov8n_face.txt
2. Increase `maxShadowTrackingAge` in tracker config
3. Lower `minDetectorConfidence` in tracker config

### If Recognition Unstable
1. Increase `max_track_embeddings` for more fusion
2. Change `fusion_mode` back to "mean" if needed
3. Enable `recognize_once_per_track=1` for stable cases