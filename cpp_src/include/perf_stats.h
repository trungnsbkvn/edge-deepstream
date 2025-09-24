#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>

namespace EdgeDeepStream {

/**
 * Thread-safe performance statistics collection and monitoring
 * with EWMA (Exponentially Weighted Moving Average) timers.
 * Matches the Python perf_stats.py implementation.
 */
class PerfStats {
public:
    // Singleton access
    static PerfStats& instance();
    
    // Disable copy/move for singleton
    PerfStats(const PerfStats&) = delete;
    PerfStats& operator=(const PerfStats&) = delete;
    PerfStats(PerfStats&&) = delete;
    PerfStats& operator=(PerfStats&&) = delete;
    
    /**
     * Initialize performance stats with configuration
     */
    void initialize();
    
    /**
     * Increment a counter by delta
     */
    void incr(const std::string& counter, int delta = 1);
    
    /**
     * Record a timer value and update EWMA
     */
    void record(const std::string& timer_key, double dt_ms);
    
    /**
     * Get a performance snapshot
     */
    struct Snapshot {
        double time;
        double elapsed;
        std::unordered_map<std::string, int> counters;
        std::unordered_map<std::string, double> timers;
        std::unordered_map<std::string, double> rates;
    };
    
    Snapshot get_snapshot();
    
    /**
     * Maybe flush statistics to disk (based on flush interval)
     */
    void maybe_flush(bool force = false);
    
    /**
     * RAII timer for measuring execution blocks
     */
    class ScopedTimer {
    public:
        explicit ScopedTimer(const std::string& timer_key);
        ~ScopedTimer();
        
        // Disable copy/move for RAII safety
        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;
        ScopedTimer(ScopedTimer&&) = delete;
        ScopedTimer& operator=(ScopedTimer&&) = delete;
        
    private:
        std::string timer_key_;
        std::chrono::high_resolution_clock::time_point start_time_;
    };
    
    /**
     * Create a scoped timer for measuring blocks
     */
    std::unique_ptr<ScopedTimer> time_block(const std::string& timer_key);
    
private:
    PerfStats();
    ~PerfStats() = default;
    
    mutable std::mutex mutex_;
    
    // State
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point last_flush_;
    std::chrono::high_resolution_clock::time_point last_print_;
    
    // Counters
    std::unordered_map<std::string, int> counters_;
    
    // EWMA timers
    std::unordered_map<std::string, double> timers_;
    
    // Configuration
    struct Config {
        double ewma_alpha = 0.2;
        double flush_interval = 2.0;
        std::string stats_path;
        double print_interval = 10.0;
        int verbose_level = 0;
    } config_;
    
    /**
     * Update EWMA for a timer
     */
    void update_ewma(const std::string& key, double ms);
    
    /**
     * Write snapshot to JSON file
     */
    bool write_json_snapshot(const Snapshot& snap);
    
    /**
     * Maybe print console summary
     */
    void maybe_print_summary(const Snapshot& snap);
    
    /**
     * Get current time as double (seconds since epoch)
     */
    static double get_current_time();
};

} // namespace EdgeDeepStream