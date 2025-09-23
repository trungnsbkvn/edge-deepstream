#!/usr/bin/env python3
"""
Performance monitoring script for the optimized facial recognition pipeline.
Tracks FPS, memory usage, detection counts, and recognition accuracy.
"""

import os
import time
import json
import psutil
from typing import Dict, List
import threading
import signal
import sys

PERF_STATS_PATH = os.environ.get('PERF_STATS_PATH', '/dev/shm/edge-deepstream/perf_stats.json')

class PipelineMonitor:
    def __init__(self, log_file="performance_log.json"):
        self.log_file = log_file
        self.running = False
        self.stats = {
            'start_time': None,
            'samples': []
        }
        
    def start_monitoring(self, interval=5):
        """Start monitoring pipeline performance."""
        self.running = True
        self.stats['start_time'] = time.time()
        
        def monitor_loop():
            while self.running:
                try:
                    sample = self.collect_sample()
                    self.stats['samples'].append(sample)
                    self.print_status(sample)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def collect_sample(self) -> Dict:
        """Collect a performance sample."""
        timestamp = time.time()
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU stats (if nvidia-ml-py is available)
        gpu_stats = self.get_gpu_stats()
        
        # Pipeline-specific stats (from log files or shared memory)
        pipeline_stats = self.get_pipeline_stats()
        
        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'gpu_stats': gpu_stats,
            'pipeline_stats': pipeline_stats
        }
    
    def get_gpu_stats(self) -> Dict:
        """Get GPU utilization and memory stats."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory,
                'memory_used_mb': mem_info.used / (1024**2),
                'memory_total_mb': mem_info.total / (1024**2),
                'memory_percent': (mem_info.used / mem_info.total) * 100
            }
        except ImportError:
            # Fallback using nvidia-smi command
            try:
                import subprocess
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    return {
                        'gpu_utilization': int(values[0]),
                        'memory_used_mb': int(values[1]),
                        'memory_total_mb': int(values[2]),
                        'memory_percent': (int(values[1]) / int(values[2])) * 100
                    }
            except Exception:
                pass
            
            return {'gpu_utilization': 0, 'memory_percent': 0}
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline-specific performance stats."""
        stats = {
            'detections_per_second': 0,
            'recognitions_per_second': 0,
            'avg_confidence': 0,
            'active_tracks': 0,
            'pgie_fps': 0,
            'sgie_fps': 0,
            'faiss_searches_per_s': 0,
            'pgie_ms': 0,
            'sgie_ms': 0,
            'faiss_ms': 0,
        }
        try:
            if os.path.exists(PERF_STATS_PATH):
                with open(PERF_STATS_PATH,'r') as f:
                    snap = json.load(f)
                rates = snap.get('rates', {})
                timers = snap.get('timers', {})
                stats.update({
                    'detections_per_second': round(rates.get('detections_per_s',0),2),
                    'recognitions_per_second': round(rates.get('recognitions_per_s',0),2),
                    'pgie_fps': round(rates.get('fps_pgie',0),2),
                    'sgie_fps': round(rates.get('fps_sgie',0),2),
                    'faiss_searches_per_s': round(rates.get('faiss_searches_per_s',0),2),
                    'pgie_ms': round(timers.get('pgie_ms_ewma',0),2),
                    'sgie_ms': round(timers.get('sgie_ms_ewma',0),2),
                    'faiss_ms': round(timers.get('faiss_ms_ewma',0),2),
                })
        except Exception:
            pass
        return stats
    
    def print_status(self, sample: Dict):
        """Print current performance status."""
        elapsed = time.time() - self.stats['start_time']
        
        print(f"\n=== Pipeline Performance Monitor (Elapsed: {elapsed:.1f}s) ===")
        print(f"CPU Usage: {sample['cpu_percent']:.1f}%")
        print(f"Memory: {sample['memory_percent']:.1f}% ({sample['memory_used_gb']:.1f} GB)")
        
        if sample['gpu_stats']['gpu_utilization'] > 0:
            gpu = sample['gpu_stats']
            print(f"GPU Usage: {gpu['gpu_utilization']}%")
            print(f"GPU Memory: {gpu['memory_percent']:.1f}% ({gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB)")
        
        pipeline = sample['pipeline_stats']
        print(f"Detections/s: {pipeline['detections_per_second']}  (PGIE FPS: {pipeline['pgie_fps']}, {pipeline['pgie_ms']} ms avg)")
        print(f"Recognitions/s: {pipeline['recognitions_per_second']}  (SGIE FPS: {pipeline['sgie_fps']}, {pipeline['sgie_ms']} ms avg)")
        print(f"FAISS searches/s: {pipeline['faiss_searches_per_s']} (avg {pipeline['faiss_ms']} ms)")
        print(f"Active Tracks: {pipeline['active_tracks']}")
    
    def stop_monitoring(self):
        """Stop monitoring and save results."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        
        # Save results
        with open(self.log_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        self.print_summary()
    
    def print_summary(self):
        """Print performance summary."""
        if not self.stats['samples']:
            return
        
        samples = self.stats['samples']
        total_time = samples[-1]['timestamp'] - samples[0]['timestamp']
        
        avg_cpu = sum(s['cpu_percent'] for s in samples) / len(samples)
        avg_memory = sum(s['memory_percent'] for s in samples) / len(samples)
        
        gpu_samples = [s['gpu_stats'] for s in samples if s['gpu_stats']['gpu_utilization'] > 0]
        if gpu_samples:
            avg_gpu = sum(s['gpu_utilization'] for s in gpu_samples) / len(gpu_samples)
            avg_gpu_mem = sum(s['memory_percent'] for s in gpu_samples) / len(gpu_samples)
        else:
            avg_gpu = avg_gpu_mem = 0
        
        print(f"\n=== Performance Summary ({total_time:.1f}s) ===")
        print(f"Average CPU Usage: {avg_cpu:.1f}%")
        print(f"Average Memory Usage: {avg_memory:.1f}%")
        if avg_gpu > 0:
            print(f"Average GPU Usage: {avg_gpu:.1f}%")
            print(f"Average GPU Memory: {avg_gpu_mem:.1f}%")
        print(f"Total Samples: {len(samples)}")
        print(f"Log saved to: {self.log_file}")

def main():
    monitor = PipelineMonitor("pipeline_performance.json")
    
    def signal_handler(sig, frame):
        print("\nShutting down monitor...")
        monitor.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting pipeline performance monitor...")
    print("Press Ctrl+C to stop monitoring and see summary")
    
    monitor.start_monitoring(interval=3)
    
    # Keep running until signal
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()