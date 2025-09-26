#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <fstream>

#include <gst/gst.h>
#include "nvbufsurface.h"

namespace EdgeDeepStream {

// Performance monitoring configuration
struct PerformanceMonitorConfig {
    bool enable_fps_monitoring = true;
    bool enable_memory_monitoring = true;
    bool enable_latency_monitoring = true;
    bool enable_gpu_monitoring = true;
    bool enable_pipeline_health = true;
    
    int fps_report_interval_ms = 2000;
    int memory_report_interval_ms = 5000;
    int latency_report_interval_ms = 3000;
    int health_check_interval_ms = 1000;
    
    bool enable_csv_logging = false;
    bool enable_json_logging = false;
    std::string log_output_path = "/tmp/deepstream_perf.log";
    
    // Thresholds for alerts
    float fps_warning_threshold = 15.0f;
    float fps_critical_threshold = 10.0f;
    float memory_warning_threshold = 80.0f;
    float memory_critical_threshold = 90.0f;
    float latency_warning_threshold_ms = 100.0f;
    float latency_critical_threshold_ms = 200.0f;
};

// Component performance metrics
struct ComponentMetrics {
    std::atomic<uint64_t> frames_processed{0};
    std::atomic<uint64_t> frames_dropped{0};
    std::atomic<uint64_t> total_processing_time_us{0};
    std::atomic<uint64_t> min_processing_time_us{UINT64_MAX};
    std::atomic<uint64_t> max_processing_time_us{0};
    
    std::atomic<float> current_fps{0.0f};
    std::atomic<float> avg_processing_time_ms{0.0f};
    std::atomic<bool> is_healthy{true};
    
    std::chrono::steady_clock::time_point last_frame_time;
    std::chrono::steady_clock::time_point component_start_time;
    
    void reset() {
        frames_processed = 0;
        frames_dropped = 0;
        total_processing_time_us = 0;
        min_processing_time_us = UINT64_MAX;
        max_processing_time_us = 0;
        current_fps = 0.0f;
        avg_processing_time_ms = 0.0f;
        is_healthy = true;
        last_frame_time = std::chrono::steady_clock::now();
        component_start_time = std::chrono::steady_clock::now();
    }
    
    void update_frame_processed(uint64_t processing_time_us) {
        frames_processed++;
        total_processing_time_us += processing_time_us;
        
        // Update min/max
        uint64_t current_min = min_processing_time_us.load();
        while (processing_time_us < current_min && 
               !min_processing_time_us.compare_exchange_weak(current_min, processing_time_us));
               
        uint64_t current_max = max_processing_time_us.load();
        while (processing_time_us > current_max && 
               !max_processing_time_us.compare_exchange_weak(current_max, processing_time_us));
        
        // Update averages
        if (frames_processed > 0) {
            avg_processing_time_ms = static_cast<float>(total_processing_time_us) / 1000.0f / frames_processed;
        }
        
        last_frame_time = std::chrono::steady_clock::now();
    }
    
    void update_fps(float fps) {
        current_fps = fps;
    }
};

// System resource metrics
struct SystemMetrics {
    std::atomic<float> cpu_usage{0.0f};
    std::atomic<float> memory_usage_percent{0.0f};
    std::atomic<size_t> memory_usage_mb{0};
    std::atomic<float> gpu_usage{0.0f};
    std::atomic<float> gpu_memory_usage_percent{0.0f};
    std::atomic<size_t> gpu_memory_usage_mb{0};
    
    std::atomic<float> disk_io_read_mb_s{0.0f};
    std::atomic<float> disk_io_write_mb_s{0.0f};
    std::atomic<float> network_rx_mb_s{0.0f};
    std::atomic<float> network_tx_mb_s{0.0f};
    
    std::chrono::steady_clock::time_point last_update;
    
    void reset() {
        cpu_usage = 0.0f;
        memory_usage_percent = 0.0f;
        memory_usage_mb = 0;
        gpu_usage = 0.0f;
        gpu_memory_usage_percent = 0.0f;
        gpu_memory_usage_mb = 0;
        disk_io_read_mb_s = 0.0f;
        disk_io_write_mb_s = 0.0f;
        network_rx_mb_s = 0.0f;
        network_tx_mb_s = 0.0f;
        last_update = std::chrono::steady_clock::now();
    }
};

// Pipeline health status
enum class PipelineHealthStatus {
    HEALTHY,
    WARNING,
    CRITICAL,
    FAILED
};

struct PipelineHealthMetrics {
    std::atomic<PipelineHealthStatus> overall_status{PipelineHealthStatus::HEALTHY};
    std::atomic<int> source_count{0};
    std::atomic<int> healthy_sources{0};
    std::atomic<int> warning_sources{0};
    std::atomic<int> failed_sources{0};
    
    std::atomic<bool> decoder_healthy{true};
    std::atomic<bool> tracker_healthy{true};
    std::atomic<bool> pgie_healthy{true};
    std::atomic<bool> sgie_healthy{true};
    std::atomic<bool> sink_healthy{true};
    
    std::atomic<uint64_t> pipeline_restarts{0};
    std::atomic<uint64_t> error_count{0};
    std::atomic<uint64_t> warning_count{0};
    
    std::chrono::steady_clock::time_point last_health_check;
    std::string last_error_message;
    mutable std::mutex error_message_mutex;
    
    void reset() {
        overall_status = PipelineHealthStatus::HEALTHY;
        source_count = 0;
        healthy_sources = 0;
        warning_sources = 0;
        failed_sources = 0;
        decoder_healthy = true;
        tracker_healthy = true;
        pgie_healthy = true;
        sgie_healthy = true;
        sink_healthy = true;
        pipeline_restarts = 0;
        error_count = 0;
        warning_count = 0;
        last_health_check = std::chrono::steady_clock::now();
        last_error_message.clear();
    }
    
    void set_error(const std::string& message) {
        std::lock_guard<std::mutex> lock(error_message_mutex);
        last_error_message = message;
        error_count++;
    }
    
    std::string get_last_error() const {
        std::lock_guard<std::mutex> lock(error_message_mutex);
        return last_error_message;
    }
};

// Latency tracking for end-to-end performance
struct LatencyTracker {
    struct LatencyMeasurement {
        uint64_t frame_id;
        std::chrono::steady_clock::time_point ingestion_time;
        std::chrono::steady_clock::time_point detection_time;
        std::chrono::steady_clock::time_point recognition_time;
        std::chrono::steady_clock::time_point output_time;
        
        float get_total_latency_ms() const {
            if (output_time > ingestion_time) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    output_time - ingestion_time);
                return duration.count() / 1000.0f;
            }
            return 0.0f;
        }
        
        float get_detection_latency_ms() const {
            if (detection_time > ingestion_time) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    detection_time - ingestion_time);
                return duration.count() / 1000.0f;
            }
            return 0.0f;
        }
        
        float get_recognition_latency_ms() const {
            if (recognition_time > detection_time) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    recognition_time - detection_time);
                return duration.count() / 1000.0f;
            }
            return 0.0f;
        }
    };
    
    std::unordered_map<uint64_t, LatencyMeasurement> active_measurements;
    mutable std::mutex measurements_mutex;
    
    std::atomic<float> avg_total_latency_ms{0.0f};
    std::atomic<float> avg_detection_latency_ms{0.0f};
    std::atomic<float> avg_recognition_latency_ms{0.0f};
    std::atomic<float> max_total_latency_ms{0.0f};
    
    void start_measurement(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(measurements_mutex);
        LatencyMeasurement measurement;
        measurement.frame_id = frame_id;
        measurement.ingestion_time = std::chrono::steady_clock::now();
        active_measurements[frame_id] = measurement;
    }
    
    void mark_detection_complete(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(measurements_mutex);
        auto it = active_measurements.find(frame_id);
        if (it != active_measurements.end()) {
            it->second.detection_time = std::chrono::steady_clock::now();
        }
    }
    
    void mark_recognition_complete(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(measurements_mutex);
        auto it = active_measurements.find(frame_id);
        if (it != active_measurements.end()) {
            it->second.recognition_time = std::chrono::steady_clock::now();
        }
    }
    
    void complete_measurement(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(measurements_mutex);
        auto it = active_measurements.find(frame_id);
        if (it != active_measurements.end()) {
            it->second.output_time = std::chrono::steady_clock::now();
            
            // Update statistics
            float total_latency = it->second.get_total_latency_ms();
            float detection_latency = it->second.get_detection_latency_ms();
            float recognition_latency = it->second.get_recognition_latency_ms();
            
            // Simple rolling average (could be improved with circular buffer)
            avg_total_latency_ms = (avg_total_latency_ms.load() * 0.9f) + (total_latency * 0.1f);
            avg_detection_latency_ms = (avg_detection_latency_ms.load() * 0.9f) + (detection_latency * 0.1f);
            avg_recognition_latency_ms = (avg_recognition_latency_ms.load() * 0.9f) + (recognition_latency * 0.1f);
            
            // Update max latency
            float current_max = max_total_latency_ms.load();
            while (total_latency > current_max && 
                   !max_total_latency_ms.compare_exchange_weak(current_max, total_latency));
            
            active_measurements.erase(it);
        }
    }
    
    void cleanup_old_measurements() {
        std::lock_guard<std::mutex> lock(measurements_mutex);
        auto now = std::chrono::steady_clock::now();
        auto it = active_measurements.begin();
        while (it != active_measurements.end()) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.ingestion_time);
            if (age.count() > 5000) {  // Remove measurements older than 5 seconds
                it = active_measurements.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// Main performance monitor class
class PerformanceMonitor {
public:
    PerformanceMonitor(const PerformanceMonitorConfig& config);
    ~PerformanceMonitor();
    
    bool initialize();
    void shutdown();
    
    // Component registration and metrics
    bool register_component(const std::string& name);
    ComponentMetrics* get_component_metrics(const std::string& name);
    void update_component_fps(const std::string& name, float fps);
    void update_component_processing_time(const std::string& name, uint64_t processing_time_us);
    void mark_component_frame_drop(const std::string& name);
    
    // System metrics
    const SystemMetrics& get_system_metrics() const { return system_metrics_; }
    void update_system_metrics();
    
    // Pipeline health
    const PipelineHealthMetrics& get_health_metrics() const { return health_metrics_; }
    void check_pipeline_health();
    void report_error(const std::string& component, const std::string& error);
    void report_warning(const std::string& component, const std::string& warning);
    
    // Latency tracking
    LatencyTracker& get_latency_tracker() { return latency_tracker_; }
    
    // Reporting and logging
    void log_performance_report() const;
    void export_metrics_csv(const std::string& filename) const;
    void export_metrics_json(const std::string& filename) const;
    
    // Configuration
    void update_config(const PerformanceMonitorConfig& config);
    const PerformanceMonitorConfig& get_config() const { return config_; }
    
    // GStreamer integration
    static gboolean bus_message_callback(GstBus* bus, GstMessage* message, gpointer user_data);
    void handle_gst_message(GstMessage* message);
    
private:
    // Monitoring threads
    void start_monitoring_threads();
    void stop_monitoring_threads();
    void fps_monitoring_thread();
    void system_monitoring_thread();
    void health_monitoring_thread();
    void latency_monitoring_thread();
    
    // Metric calculation helpers
    float calculate_component_fps(const std::string& name);
    void update_health_status();
    bool check_component_health(const std::string& name, const ComponentMetrics& metrics);
    
    // Logging helpers
    void write_to_log_file(const std::string& message);
    std::string format_performance_summary() const;
    
    PerformanceMonitorConfig config_;
    bool initialized_;
    
    // Component metrics storage
    std::unordered_map<std::string, std::unique_ptr<ComponentMetrics>> component_metrics_;
    mutable std::mutex component_metrics_mutex_;
    
    // System and health metrics
    SystemMetrics system_metrics_;
    PipelineHealthMetrics health_metrics_;
    LatencyTracker latency_tracker_;
    
    // Monitoring threads
    std::vector<std::thread> monitoring_threads_;
    std::atomic<bool> monitoring_active_{false};
    
    // Logging
    std::ofstream log_file_;
    mutable std::mutex log_file_mutex_;
    
    // Performance history for trend analysis
    struct PerformanceSnapshot {
        std::chrono::steady_clock::time_point timestamp;
        float overall_fps;
        float cpu_usage;
        float memory_usage;
        float gpu_usage;
        PipelineHealthStatus health_status;
    };
    
    std::vector<PerformanceSnapshot> performance_history_;
    mutable std::mutex performance_history_mutex_;
    static const size_t MAX_HISTORY_SIZE = 1000;
};

// Utility classes for performance monitoring integration
class PerformanceTimer {
public:
    PerformanceTimer(PerformanceMonitor* monitor, const std::string& component);
    ~PerformanceTimer();
    
private:
    PerformanceMonitor* monitor_;
    std::string component_;
    std::chrono::steady_clock::time_point start_time_;
};

class FrameLatencyTracker {
public:
    FrameLatencyTracker(PerformanceMonitor* monitor, uint64_t frame_id);
    ~FrameLatencyTracker();
    
    void mark_detection_complete();
    void mark_recognition_complete();
    
private:
    PerformanceMonitor* monitor_;
    uint64_t frame_id_;
    bool completed_;
};

// RAII helpers for automatic performance monitoring
#define PERF_TIMER(monitor, component) \
    PerformanceTimer _perf_timer(monitor, component)

#define LATENCY_TRACKER(monitor, frame_id) \
    FrameLatencyTracker _latency_tracker(monitor, frame_id)

} // namespace EdgeDeepStream