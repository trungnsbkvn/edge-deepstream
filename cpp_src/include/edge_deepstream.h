#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <algorithm>

// DeepStream headers
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"
#include "nvbufsurface.h"

// OpenCV
#include <opencv2/opencv.hpp>

// Project headers
#include "config_parser.h"
#include "status_codes.h"
#include "env_utils.h"
#include "faiss_index.h"
#include "placeholders.h"

namespace EdgeDeepStream {

// Forward declarations
class Pipeline;
class EventSender;
class MQTTListener;

// Global configuration structure
struct Config {
    std::map<std::string, std::map<std::string, std::string>> sections;
    
    template<typename T>
    T get(const std::string& section, const std::string& key, const T& defaultValue = T{}) const {
        // Default implementation for unsupported types
        return defaultValue;
    }
    
    bool has_section(const std::string& section) const {
        return sections.find(section) != sections.end();
    }
    
    bool has_key(const std::string& section, const std::string& key) const {
        if (!has_section(section)) {
            return false;
        }
        return sections.at(section).find(key) != sections.at(section).end();
    }
};

// Template specializations for Config::get
template<>
inline std::string Config::get<std::string>(const std::string& section, const std::string& key, const std::string& defaultValue) const {
    if (!has_section(section) || !has_key(section, key)) {
        return defaultValue;
    }
    return sections.at(section).at(key);
}

template<>
inline int Config::get<int>(const std::string& section, const std::string& key, const int& defaultValue) const {
    if (!has_section(section) || !has_key(section, key)) {
        return defaultValue;
    }
    try {
        return std::stoi(sections.at(section).at(key));
    } catch (...) {
        return defaultValue;
    }
}

template<>
inline bool Config::get<bool>(const std::string& section, const std::string& key, const bool& defaultValue) const {
    if (!has_section(section) || !has_key(section, key)) {
        return defaultValue;
    }
    std::string value = sections.at(section).at(key);
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return (value == "1" || value == "true" || value == "yes" || value == "on");
}

// Source information
struct SourceInfo {
    std::string id;
    std::string uri;
    bool is_rtsp;
    GstElement* source_bin;
    GstPad* sink_pad;
    int index;
};

// Application state
class Application {
public:
    Application();
    ~Application();
    
    bool initialize(const std::string& config_path);
    bool run(int duration_ms = -1);  // -1 for infinite
    void shutdown();
    
    // Configuration access
    const Config& get_config() const { return config_; }
    
    // Pipeline access
    Pipeline* get_pipeline() { return pipeline_.get(); }
    
    // MQTT and events
    EventSender* get_event_sender() { return event_sender_.get(); }
    MQTTListener* get_mqtt_listener() { return mqtt_listener_.get(); }
    
    // Face recognition
    FaceIndex* get_face_index() { return face_index_.get(); }
    
    // Bus message handler
    gboolean bus_message_handler(GstBus* bus, GstMessage* msg);

private:
    bool setup_pipeline();
    bool setup_recognition();
    bool setup_mqtt();
    
    Config config_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<EventSender> event_sender_;
    std::unique_ptr<MQTTListener> mqtt_listener_;
    std::unique_ptr<FaceIndex> face_index_;
    
    std::atomic<bool> running_;
    GMainLoop* loop_;
};

// Global realtime flag (matches Python implementation)
extern bool REALTIME_DROP;

// Utility functions
std::vector<std::string> load_known_faces(const std::string& path);
bool setup_gstreamer_plugins();

} // namespace EdgeDeepStream