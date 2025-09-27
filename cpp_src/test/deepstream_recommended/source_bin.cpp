#include "source_bin.h"
#include <iostream>
#include <algorithm>
#include <gst/gst.h>
#include <string>

namespace EdgeDeepStream {

SourceBin::SourceBin(int index, const std::string& uri)
    : index_(index), uri_(uri), is_rtsp_(uri.find("rtsp://") == 0),
      bin_(nullptr), source_element_(nullptr) {}

SourceBin::~SourceBin() {}

bool SourceBin::create() {
    std::string bin_name = "source-bin-" + std::to_string(index_);
    bin_ = gst_bin_new(bin_name.c_str());
    if (!bin_) {
        std::cerr << "Unable to create source bin" << std::endl;
        return false;
    }
    return create_uri_decode_source();
}

bool SourceBin::set_state(GstState state) {
    if (!bin_) return false;
    std::cout << "Setting source " << index_ << " to " << gst_element_state_get_name(state) << std::endl;
    GstStateChangeReturn ret = gst_element_set_state(bin_, state);
    return ret != GST_STATE_CHANGE_FAILURE;
}

bool SourceBin::create_uri_decode_source() {
    std::cout << "Creating URI decode source for " << uri_ << std::endl;
    std::string decode_name = "uri-decode-bin-" + std::to_string(index_);
    source_element_ = gst_element_factory_make("uridecodebin", decode_name.c_str());
    if (!source_element_) {
        std::cerr << "Unable to create uri decode bin" << std::endl;
        return false;
    }
    g_object_set(source_element_, "uri", uri_.c_str(), NULL);
    g_signal_connect(source_element_, "pad-added", G_CALLBACK(cb_newpad), this);
    g_signal_connect(source_element_, "source-setup", G_CALLBACK(source_setup), this);
    gst_bin_add(GST_BIN(bin_), source_element_);
    GstPad* ghost_pad = gst_ghost_pad_new_no_target("src", GST_PAD_SRC);
    gst_element_add_pad(bin_, ghost_pad);
    std::cout << "URI decode source created successfully" << std::endl;
    return true;
}

void SourceBin::cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data) {
    SourceBin* source_bin = static_cast<SourceBin*>(data);
    std::cout << "[cb_newpad] PAD-ADDED for source " << source_bin->index_ << std::endl;
    GstCaps* caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps) {
        std::cout << "[cb_newpad] No caps on pad!" << std::endl;
        return;
    }
    gchar* caps_str = gst_caps_to_string(caps);
    std::cout << "[cb_newpad] Pad caps: " << (caps_str ? caps_str : "(null)") << std::endl;
    g_free(caps_str);
    GstStructure* structure = gst_caps_get_structure(caps, 0);
    const gchar* name = gst_structure_get_name(structure);
    std::cout << "[cb_newpad] Structure name: " << (name ? name : "(null)") << std::endl;
    if (g_strrstr(name, "video")) {
        GstCapsFeatures* features = gst_caps_get_features(caps, 0);
        if (gst_caps_features_contains(features, "memory:NVMM")) {
            std::cout << "[cb_newpad] Hardware decoder detected (NVMM) - linking pad" << std::endl;
            GstPad* ghost_pad = gst_element_get_static_pad(GST_ELEMENT(source_bin->bin_), "src");
            if (ghost_pad && GST_IS_GHOST_PAD(ghost_pad)) {
                gst_ghost_pad_set_target(GST_GHOST_PAD(ghost_pad), decoder_src_pad);
                gst_object_unref(ghost_pad);
                std::cout << "[cb_newpad] Source " << source_bin->index_ << " connected successfully" << std::endl;
            }
        } else {
            std::cout << "[cb_newpad] Pad does NOT have NVMM (hardware) memory. Ignoring for performance." << std::endl;
        }
    } else {
        std::cout << "[cb_newpad] Non-video pad detected. Ignoring." << std::endl;
    }
    gst_caps_unref(caps);
}

void SourceBin::source_setup(GstElement* uridecodebin, GstElement* source, gpointer user_data) {
    SourceBin* source_bin = static_cast<SourceBin*>(user_data);
    if (g_str_has_prefix(GST_ELEMENT_NAME(source), "rtspsrc")) {
        std::cout << "Configuring RTSP source " << source_bin->index_ << " for multi-stream performance" << std::endl;
        g_object_set(source,
                     "latency", 50,
                     "drop-on-latency", TRUE,
                     "do-retransmission", FALSE,
                     "protocols", 4,
                     "buffer-mode", 1,
                     NULL);
        std::cout << "RTSP source " << source_bin->index_ << " optimized for real-time processing" << std::endl;
    }
}

} // namespace EdgeDeepStream
