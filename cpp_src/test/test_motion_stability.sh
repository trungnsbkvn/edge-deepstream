#!/bin/bash

# Motion Stability Test and Validation Script
# Tests the optimized pipeline for motion scenarios

echo "=== Motion Stability Validation ==="
echo "Testing pipeline performance for walking/motion scenarios..."

# Test duration in seconds
TEST_DURATION=60
PERFORMANCE_LOG="/dev/shm/edge-deepstream/motion_test_results.txt"

# Clean up any existing performance logs
rm -f $PERFORMANCE_LOG
mkdir -p /dev/shm/edge-deepstream

echo "Starting motion-optimized pipeline test..."
echo "Test duration: ${TEST_DURATION} seconds"
echo "Performance log: ${PERFORMANCE_LOG}"

# Start performance monitoring in background
{
    echo "=== Motion Stability Test Results ===" > $PERFORMANCE_LOG
    echo "Start time: $(date)" >> $PERFORMANCE_LOG
    echo "" >> $PERFORMANCE_LOG
    
    # Monitor for the test duration
    for i in $(seq 1 $TEST_DURATION); do
        if [ -f "/dev/shm/edge-deepstream/perf_stats_motion.json" ]; then
            echo "Sample $i:" >> $PERFORMANCE_LOG
            cat /dev/shm/edge-deepstream/perf_stats_motion.json >> $PERFORMANCE_LOG
            echo "" >> $PERFORMANCE_LOG
        fi
        sleep 1
    done
    
    echo "=== Test Summary ===" >> $PERFORMANCE_LOG
    echo "End time: $(date)" >> $PERFORMANCE_LOG
    
    # Calculate improvements
    if [ -f "/dev/shm/edge-deepstream/perf_stats_motion.json" ]; then
        echo "" >> $PERFORMANCE_LOG
        echo "Final Performance Metrics:" >> $PERFORMANCE_LOG
        cat /dev/shm/edge-deepstream/perf_stats_motion.json | python3 -c "
import sys, json
data = json.load(sys.stdin)
rates = data.get('rates', {})
timers = data.get('timers', {})
print(f'PGIE FPS: {rates.get(\"fps_pgie\", 0):.2f}')
print(f'SGIE FPS: {rates.get(\"fps_sgie\", 0):.2f}')
print(f'Detection Rate: {rates.get(\"detections_per_s\", 0):.2f}/s')
print(f'Recognition Rate: {rates.get(\"recognitions_per_s\", 0):.2f}/s')
print(f'PGIE Processing: {timers.get(\"pgie_ms_ewma\", 0):.2f}ms')
print(f'SGIE Processing: {timers.get(\"sgie_ms_ewma\", 0):.2f}ms')

# Performance assessment
pgie_fps = rates.get('fps_pgie', 0)
sgie_fps = rates.get('sgie_fps', 0)
pgie_ms = timers.get('pgie_ms_ewma', 0)
sgie_ms = timers.get('sgie_ms_ewma', 0)

print('\\n=== MOTION STABILITY ASSESSMENT ===')
if pgie_fps > 8:
    print('✅ PGIE FPS: EXCELLENT (>8 FPS)')
elif pgie_fps > 5:
    print('✅ PGIE FPS: GOOD (5-8 FPS)')
elif pgie_fps > 3:
    print('⚠️  PGIE FPS: ACCEPTABLE (3-5 FPS)')
else:
    print('❌ PGIE FPS: NEEDS IMPROVEMENT (<3 FPS)')

if sgie_fps > 5:
    print('✅ SGIE FPS: EXCELLENT (>5 FPS)')
elif sgie_fps > 3:
    print('✅ SGIE FPS: GOOD (3-5 FPS)')
elif sgie_fps > 1:
    print('⚠️  SGIE FPS: ACCEPTABLE (1-3 FPS)')
else:
    print('❌ SGIE FPS: NEEDS IMPROVEMENT (<1 FPS)')

if pgie_ms < 1:
    print('✅ PGIE Latency: EXCELLENT (<1ms)')
elif pgie_ms < 2:
    print('✅ PGIE Latency: GOOD (1-2ms)')
else:
    print('⚠️  PGIE Latency: HIGH (>2ms)')

if sgie_ms < 3:
    print('✅ SGIE Latency: EXCELLENT (<3ms)')
elif sgie_ms < 5:
    print('✅ SGIE Latency: GOOD (3-5ms)')
elif sgie_ms < 10:
    print('⚠️  SGIE Latency: ACCEPTABLE (5-10ms)')
else:
    print('❌ SGIE Latency: HIGH (>10ms)')

# Overall motion stability score
score = 0
if pgie_fps > 5: score += 30
elif pgie_fps > 3: score += 20
elif pgie_fps > 1: score += 10

if sgie_fps > 3: score += 25
elif sgie_fps > 1: score += 15
elif sgie_fps > 0.5: score += 10

if pgie_ms < 2: score += 25
elif pgie_ms < 5: score += 15
elif pgie_ms < 10: score += 10

if sgie_ms < 5: score += 20
elif sgie_ms < 10: score += 15
elif sgie_ms < 20: score += 10

print(f'\\nMotion Stability Score: {score}/100')
if score >= 80:
    print('🎯 EXCELLENT: Ready for production motion scenarios')
elif score >= 60:
    print('✅ GOOD: Suitable for most motion scenarios')
elif score >= 40:
    print('⚠️  ACCEPTABLE: May struggle with fast motion')
else:
    print('❌ POOR: Needs further optimization')
" >> $PERFORMANCE_LOG
    fi
} &

MONITOR_PID=$!

# Start the motion-optimized pipeline
./run_motion_optimized.sh &
PIPELINE_PID=$!

# Wait for test duration
sleep $TEST_DURATION

# Stop the pipeline
kill $PIPELINE_PID 2>/dev/null
wait $PIPELINE_PID 2>/dev/null

# Wait for monitoring to complete
wait $MONITOR_PID 2>/dev/null

echo ""
echo "=== Test completed! ==="
echo "Results saved to: $PERFORMANCE_LOG"
echo ""
echo "Performance Summary:"
if [ -f "$PERFORMANCE_LOG" ]; then
    tail -20 "$PERFORMANCE_LOG"
else
    echo "Performance log not found. Pipeline may not have started properly."
fi

echo ""
echo "Motion optimization recommendations:"
echo "1. For walking scenarios: Current settings should work well"
echo "2. For running/fast motion: Consider further reducing resolution"
echo "3. For crowded scenes: Increase detection thresholds"
echo "4. For better recognition: Enable GPU acceleration if available"