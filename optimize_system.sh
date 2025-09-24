#!/bin/bash

# System Process Utilization Optimization Script
# Maximizes active processes and reduces system overhead

echo "=========================================="
echo "Process Utilization Optimization Script"
echo "=========================================="

# Check current system state
echo "Current system state:"
echo "Load average: $(uptime | cut -d, -f3-5)"
echo "Active processes: $(ps aux | wc -l)"
echo "Total threads: $(ps -eLf | wc -l)"
echo "Memory usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"

echo ""
echo "Applying optimizations..."

# 1. Kernel scheduler optimizations
echo "1. Optimizing kernel scheduler..."
echo 1 | sudo tee /proc/sys/kernel/sched_autogroup_enabled > /dev/null
echo 0 | sudo tee /proc/sys/kernel/sched_child_runs_first > /dev/null

# 2. Memory management optimizations
echo "2. Optimizing memory management..."
echo 1 | sudo tee /proc/sys/vm/overcommit_memory > /dev/null
echo 50 | sudo tee /proc/sys/vm/overcommit_ratio > /dev/null
echo 10 | sudo tee /proc/sys/vm/swappiness > /dev/null
echo 0 | sudo tee /proc/sys/vm/zone_reclaim_mode > /dev/null

# 3. Network optimizations for RTSP streams
echo "3. Optimizing network for RTSP..."
echo 262144 | sudo tee /proc/sys/net/core/rmem_max > /dev/null
echo 262144 | sudo tee /proc/sys/net/core/wmem_max > /dev/null
echo '4096 87380 262144' | sudo tee /proc/sys/net/ipv4/tcp_rmem > /dev/null
echo '4096 65536 262144' | sudo tee /proc/sys/net/ipv4/tcp_wmem > /dev/null

# 4. File system optimizations
echo "4. Optimizing file system..."
echo 65536 | sudo tee /proc/sys/fs/file-max > /dev/null
echo '1024 65536' | sudo tee /proc/sys/net/ipv4/ip_local_port_range > /dev/null

# 5. Process limits optimization
echo "5. Optimizing process limits..."
# Temporarily increase limits for current session
ulimit -n 8192      # File descriptors
ulimit -u 65536     # Max user processes
ulimit -l unlimited # Memory lock
ulimit -s 16384     # Stack size

# 6. CPU frequency scaling
echo "6. Setting CPU performance mode..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -w "$cpu" ]; then
        echo performance | sudo tee "$cpu" > /dev/null 2>&1
    fi
done

# 7. Disable unnecessary services temporarily (optional)
echo "7. Optimizing system services..."
# Stop non-essential services to free up processes
sudo systemctl stop snapd.service 2>/dev/null || true
sudo systemctl stop ModemManager.service 2>/dev/null || true
sudo systemctl stop bluetooth.service 2>/dev/null || true

# 8. IRQ affinity optimization for multi-core
echo "8. Optimizing interrupt handling..."
# Set network interrupts to specific CPUs
for irq in $(grep eth /proc/interrupts | cut -d: -f1 | tr -d ' '); do
    echo 2 | sudo tee /proc/irq/$irq/smp_affinity > /dev/null 2>&1
done

# 9. Create optimized cgroup for DeepStream
echo "9. Setting up process groups..."
if [ -d /sys/fs/cgroup/cpu ]; then
    sudo mkdir -p /sys/fs/cgroup/cpu/deepstream 2>/dev/null || true
    echo 950000 | sudo tee /sys/fs/cgroup/cpu/deepstream/cpu.cfs_quota_us > /dev/null 2>&1
    echo 100000 | sudo tee /sys/fs/cgroup/cpu/deepstream/cpu.cfs_period_us > /dev/null 2>&1
fi

echo ""
echo "Optimization complete!"
echo ""
echo "New system state:"
echo "Available file descriptors: $(ulimit -n)"
echo "Max user processes: $(ulimit -u)"
echo "CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
echo ""
echo "Recommendations:"
echo "1. Use 'run_maxperf.sh' for optimal DeepStream performance"
echo "2. Monitor with: watch -n 1 'ps aux --sort=-%cpu | head -10'"
echo "3. Check memory: watch -n 1 'free -h'"
echo "4. Monitor load: watch -n 1 'uptime'"
echo ""
echo "To make these settings permanent, add them to /etc/sysctl.conf"
echo "and /etc/security/limits.conf"