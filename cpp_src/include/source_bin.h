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
    bool create_uri_decode_source();
    
    // Essential callbacks for hardware decoder
    static void cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data);
    static void source_setup(GstElement* uridecodebin, GstElement* source, gpointer user_data);
    
    // Core properties
    int index_;
    std::string uri_;
    bool is_rtsp_;
    
    // GStreamer elements
    GstElement* bin_;
    GstElement* source_element_;  // uridecodebin only
};

} // namespace EdgeDeepStream