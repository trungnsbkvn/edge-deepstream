#pragma once

#include "edge_deepstream.h"
#include <functional>

namespace EdgeDeepStream {

// Pipeline callback types
using PadAddedCallback = std::function<void(GstElement*, GstPad*)>;
using BusMessageCallback = std::function<gboolean(GstBus*, GstMessage*)>;

class Pipeline {
public:
    Pipeline();
    ~Pipeline();
    
    bool create(const Config& config);
    bool start();
    bool stop();
    void destroy();
    
    // Pipeline control
    bool set_state(GstState state);
    GstState get_state();
    
    // Source management
    bool add_source(const SourceInfo& source);
    bool remove_source(const std::string& source_id);
    
    // Element access
    GstElement* get_pipeline() { return pipeline_; }
    GstElement* get_streammux() { return streammux_; }
    GstElement* get_pgie() { return pgie_; }
    GstElement* get_sgie() { return sgie_; }
    GstElement* get_tracker() { return tracker_; }
    GstElement* get_tiler() { return tiler_; }
    GstElement* get_sink() { return sink_; }
    
    // Callbacks
    void set_bus_callback(BusMessageCallback callback);
    
    // Statistics
    void update_performance_stats();
    
private:
    bool create_elements(const Config& config);
    bool link_elements();
    bool setup_element_properties(const Config& config);
    void setup_queue_properties();
    
    // Element creation helpers
    GstElement* create_streammux(const Config& config);
    GstElement* create_pgie(const Config& config);
    GstElement* create_sgie(const Config& config);
    GstElement* create_tracker(const Config& config);
    GstElement* create_tiler(const Config& config);
    GstElement* create_nvvidconv(const Config& config);
    GstElement* create_sink(const Config& config);
    GstElement* create_osd(const Config& config);
    
    // Pipeline elements
    GstElement* pipeline_;
    GstElement* streammux_;
    GstElement* pgie_;
    GstElement* sgie_;
    GstElement* tracker_;
    GstElement* tiler_;
    GstElement* nvvidconv_;  // nvvideoconvert element like Python version
    GstElement* osd_;
    GstElement* sink_;
    GstElement* queue1_;
    GstElement* queue2_;
    GstElement* queue3_;
    GstElement* queue4_;
    GstElement* queue5_;
    GstElement* queue6_;
    GstElement* queue7_;
    
    // Bus and callbacks
    GstBus* bus_;
    guint bus_watch_id_;
    BusMessageCallback bus_callback_;
    
    // Configuration
    Config config_;
    
    // Sources
    std::map<std::string, SourceInfo> sources_;
    int next_source_index_;
    
    // Performance tracking
    std::chrono::steady_clock::time_point start_time_;
    uint64_t frame_count_;
    
    // Static callback wrappers
    static gboolean bus_call_wrapper(GstBus* bus, GstMessage* msg, gpointer data);
};

} // namespace EdgeDeepStream