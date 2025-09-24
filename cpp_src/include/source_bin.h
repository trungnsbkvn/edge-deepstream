#pragma once

#include "edge_deepstream.h"

namespace EdgeDeepStream {

class SourceBin {
public:
    SourceBin(int index, const std::string& uri);
    ~SourceBin();
    
    bool create();
    GstElement* get_bin() { return bin_; }
    
    // Source properties
    int get_index() const { return index_; }
    const std::string& get_uri() const { return uri_; }
    bool is_rtsp() const { return is_rtsp_; }
    
    // State management
    bool set_state(GstState state);
    
private:
    bool create_rtsp_source();
    bool create_uri_decode_source();
    
    // RTSP-specific methods
    bool setup_rtsp_elements();
    bool setup_rtsp_properties(GstElement* rtspsrc);
    void setup_rtsp_queues();
    
    // Callbacks
    static void cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data);
    static void decodebin_child_added(GstChildProxy* child_proxy, GObject* object, 
                                    const gchar* name, gpointer user_data);
    static void rtsp_pad_added(GstElement* src, GstPad* pad, gpointer user_data);
    
    // Dynamic element management for RTSP
    struct DynamicElements {
        GstElement* depay = nullptr;
        GstElement* parse = nullptr;
        GstElement* decoder = nullptr;
        GstElement* capsfilter = nullptr;
    };
    
    int index_;
    std::string uri_;
    bool is_rtsp_;
    
    GstElement* bin_;
    GstElement* source_element_;  // rtspsrc or uridecodebin
    GstElement* queue_pre_;
    GstElement* queue_post_;
    
    DynamicElements dynamic_elements_;
    bool pipeline_linked_;
    
    // Configuration
    struct RTSPConfig {
        int latency = 150;
        bool tcp_mode = false;
        bool do_rtcp = false;
        bool do_retransmission = false;
        bool ntp_sync = false;
        bool drop_on_latency = false;
        int timeout_us = 5000000;
        int retry_count = 3;
        int tcp_timeout_us = 5000000;
        std::string user_agent = "DeepStream/1.0";
        int buffer_mode = 1;  // LOW_LATENCY
    };
    
    RTSPConfig rtsp_config_;
};

} // namespace EdgeDeepStream