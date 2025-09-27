#pragma once
#include <gst/gst.h>
#include <string>

namespace EdgeDeepStream {
class SourceBin {
public:
    SourceBin(int index, const std::string& uri);
    ~SourceBin();
    bool create();
    bool set_state(GstState state);
    GstElement* get_bin() const { return bin_; }
    int get_index() const { return index_; }
    const std::string& get_uri() const { return uri_; }
    bool is_rtsp() const { return is_rtsp_; }
private:
    bool create_uri_decode_source();
    static void cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data);
    static void source_setup(GstElement* uridecodebin, GstElement* source, gpointer user_data);
    int index_;
    std::string uri_;
    bool is_rtsp_;
    GstElement* bin_;
    GstElement* source_element_;
};
} // namespace EdgeDeepStream
