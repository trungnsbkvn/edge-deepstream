#include "perf_stats.h"
#include "env_utils.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace EdgeDeepStream {

PerfStats& PerfStats::instance() {
    static PerfStats instance;
    return instance;
}

PerfStats::PerfStats()
    : start_time_(std::chrono::high_resolution_clock::now())
    , last_flush_(start_time_)
    , last_print_(start_time_)
{
    // Initialize standard counters
    counters_["frames_pgie"] = 0;
    counters_["detections"] = 0;
    counters_["frames_sgie"] = 0;
    counters_["recognition_attempts"] = 0;
    counters_["recognition_matches"] = 0;
    counters_["faiss_searches"] = 0;
    
    // Initialize standard timers
    timers_["pgie_ms_ewma"] = 0.0;
    timers_["sgie_ms_ewma"] = 0.0;
    timers_["faiss_ms_ewma"] = 0.0;
    timers_["embed_ms_ewma"] = 0.0;
}

void PerfStats::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Load configuration from environment
    config_.ewma_alpha = EnvUtils::env_float("PERF_EWMA_ALPHA", 0.2);
    config_.flush_interval = EnvUtils::env_float("PERF_FLUSH_INTERVAL", 2.0);
    config_.stats_path = EnvUtils::env_str("PERF_STATS_PATH", "/dev/shm/edge-deepstream/perf_stats.json");
    config_.print_interval = EnvUtils::env_float("PERF_PRINT_INTERVAL", 10.0);
    config_.verbose_level = EnvUtils::env_int("PERF_VERBOSE", 0);
    
    // Create directory for stats file
    std::filesystem::path stats_dir = std::filesystem::path(config_.stats_path).parent_path();
    try {
        std::filesystem::create_directories(stats_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create perf stats directory: " << e.what() << std::endl;
    }
    
    if (config_.verbose_level > 0) {
        std::cout << "PerfStats initialized: path=" << config_.stats_path 
                  << " verbose=" << config_.verbose_level << std::endl;
    }
}

void PerfStats::incr(const std::string& counter, int delta) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_[counter] += delta;
}

void PerfStats::record(const std::string& timer_key, double dt_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    update_ewma(timer_key, dt_ms);
}

PerfStats::Snapshot PerfStats::get_snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_duration = now - start_time_;
    double elapsed = std::chrono::duration<double>(elapsed_duration).count();
    elapsed = std::max(elapsed, 1e-6);  // Avoid division by zero
    
    Snapshot snap;
    snap.time = get_current_time();
    snap.elapsed = elapsed;
    snap.counters = counters_;
    snap.timers = timers_;
    
    // Calculate rates per second
    snap.rates["fps_pgie"] = counters_.at("frames_pgie") / elapsed;
    snap.rates["fps_sgie"] = counters_.at("frames_sgie") / elapsed;
    snap.rates["detections_per_s"] = counters_.at("detections") / elapsed;
    snap.rates["recognitions_per_s"] = counters_.at("recognition_matches") / elapsed;
    snap.rates["faiss_searches_per_s"] = counters_.at("faiss_searches") / elapsed;
    
    return snap;
}

void PerfStats::maybe_flush(bool force) {
    auto now = std::chrono::high_resolution_clock::now();
    
    bool should_flush = force;
    if (!should_flush) {
        auto elapsed = std::chrono::duration<double>(now - last_flush_).count();
        should_flush = elapsed >= config_.flush_interval;
    }
    
    if (should_flush) {
        auto snap = get_snapshot();
        
        // Write to disk
        if (write_json_snapshot(snap)) {
            std::lock_guard<std::mutex> lock(mutex_);
            last_flush_ = now;
        }
        
        // Maybe print console summary
        maybe_print_summary(snap);
    }
}

std::unique_ptr<PerfStats::ScopedTimer> PerfStats::time_block(const std::string& timer_key) {
    return std::make_unique<ScopedTimer>(timer_key);
}

PerfStats::ScopedTimer::ScopedTimer(const std::string& timer_key)
    : timer_key_(timer_key)
    , start_time_(std::chrono::high_resolution_clock::now())
{
}

PerfStats::ScopedTimer::~ScopedTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time_);
    PerfStats::instance().record(timer_key_, duration.count());
}

void PerfStats::update_ewma(const std::string& key, double ms) {
    double& current = timers_[key];
    if (current <= 0.0) {
        current = ms;
    } else {
        current = (1.0 - config_.ewma_alpha) * current + config_.ewma_alpha * ms;
    }
}

bool PerfStats::write_json_snapshot(const Snapshot& snap) {
    try {
        std::ostringstream json;
        json << std::fixed << std::setprecision(6);
        json << "{";
        json << "\"time\":" << snap.time << ",";
        json << "\"elapsed\":" << snap.elapsed << ",";
        
        // Counters
        json << "\"counters\":{";
        bool first = true;
        for (const auto& [key, value] : snap.counters) {
            if (!first) json << ",";
            json << "\"" << key << "\":" << value;
            first = false;
        }
        json << "},";
        
        // Timers
        json << "\"timers\":{";
        first = true;
        for (const auto& [key, value] : snap.timers) {
            if (!first) json << ",";
            json << "\"" << key << "\":" << value;
            first = false;
        }
        json << "},";
        
        // Rates
        json << "\"rates\":{";
        first = true;
        for (const auto& [key, value] : snap.rates) {
            if (!first) json << ",";
            json << "\"" << key << "\":" << value;
            first = false;
        }
        json << "}";
        json << "}";
        
        // Write to temporary file, then rename (atomic)
        std::string temp_path = config_.stats_path + ".tmp";
        std::ofstream file(temp_path);
        if (!file) {
            return false;
        }
        
        file << json.str();
        file.close();
        
        if (file.good()) {
            std::filesystem::rename(temp_path, config_.stats_path);
            return true;
        }
        
    } catch (const std::exception& e) {
        if (config_.verbose_level > 0) {
            std::cerr << "Failed to write perf stats: " << e.what() << std::endl;
        }
    }
    
    return false;
}

void PerfStats::maybe_print_summary(const Snapshot& snap) {
    if (config_.verbose_level < 1) {
        return;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_print_).count();
    
    if (elapsed >= config_.print_interval) {
        const auto& r = snap.rates;
        const auto& t = snap.timers;
        
        std::cout << std::fixed << std::setprecision(2)
                  << "[PERF] pgie_fps=" << r.at("fps_pgie")
                  << " sgie_fps=" << r.at("fps_sgie") 
                  << " det/s=" << r.at("detections_per_s")
                  << " rec/s=" << r.at("recognitions_per_s")
                  << " faiss_ms=" << t.at("faiss_ms_ewma") << std::endl;
        
        std::lock_guard<std::mutex> lock(mutex_);
        last_print_ = now;
    }
}

double PerfStats::get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

} // namespace EdgeDeepStream