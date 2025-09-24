#include "source_bin.h"
#include "env_utils.h"
#include <iostream>
#include <algorithm>

namespace EdgeDeepStream {

SourceBin::SourceBin(int index, const std::string& uri) 
    : index_(index), uri_(uri), is_rtsp_(uri.find("rtsp://") == 0),
      bin_(nullptr), source_element_(nullptr), queue_pre_(nullptr), queue_post_(nullptr),
      pipeline_linked_(false) {
    
    // Initialize RTSP configuration from environment
    rtsp_config_.latency = EnvUtils::env_int("DS_RTSP_LATENCY", 150);
    rtsp_config_.tcp_mode = EnvUtils::env_bool("DS_RTSP_TCP", false).value_or(false);
    rtsp_config_.do_rtcp = EnvUtils::env_bool("DS_RTSP_DO_RTCP", false).value_or(false);
    rtsp_config_.do_retransmission = EnvUtils::env_bool("DS_RTSP_RETRANS", false).value_or(false);
    rtsp_config_.ntp_sync = EnvUtils::env_bool("DS_RTSP_NTP_SYNC", false).value_or(false);
    rtsp_config_.drop_on_latency = EnvUtils::env_bool("DS_RTSP_DROP_ON_LATENCY").value_or(false);
    rtsp_config_.timeout_us = EnvUtils::env_int("DS_RTSP_TIMEOUT_US", 5000000);
    rtsp_config_.retry_count = EnvUtils::env_int("DS_RTSP_RETRY", 3);
    rtsp_config_.tcp_timeout_us = EnvUtils::env_int("DS_RTSP_TCP_TIMEOUT_US", 5000000);
    rtsp_config_.user_agent = EnvUtils::env_str("DS_RTSP_USER_AGENT", "DeepStream/1.0");
    rtsp_config_.buffer_mode = EnvUtils::env_int("DS_RTSP_BUFFER_MODE", 1);
}

SourceBin::~SourceBin() {
    // Elements will be cleaned up when the bin is destroyed
}

bool SourceBin::create() {
    // Create the bin
    std::string bin_name = "source-bin-" + std::to_string(index_);
    bin_ = gst_bin_new(bin_name.c_str());
    if (!bin_) {
        std::cerr << "Unable to create source bin" << std::endl;
        return false;
    }
    
    if (is_rtsp_) {
        return create_rtsp_source();
    } else {
        return create_uri_decode_source();
    }
}

bool SourceBin::set_state(GstState state) {
    if (!bin_) {
        return false;
    }
    
    GstStateChangeReturn ret = gst_element_set_state(bin_, state);
    return ret != GST_STATE_CHANGE_FAILURE;
}

bool SourceBin::create_rtsp_source() {
    try {
        std::cout << "[RTSP] Creating RTSP source index=" << index_ << " uri=" << uri_ << std::endl;
        
        // Create rtspsrc
        std::string rtspsrc_name = "rtspsrc-" + std::to_string(index_);
        source_element_ = gst_element_factory_make("rtspsrc", rtspsrc_name.c_str());
        if (!source_element_) {
            std::cerr << "Failed to create rtspsrc element" << std::endl;
            return false;
        }
        
        // Set URI
        g_object_set(source_element_, "location", uri_.c_str(), NULL);
        
        // Setup RTSP properties
        if (!setup_rtsp_properties(source_element_)) {
            std::cerr << "Failed to setup RTSP properties" << std::endl;
            return false;
        }
        
        // Create queues
        setup_rtsp_queues();
        
        // Add initial elements to bin
        gst_bin_add_many(GST_BIN(bin_), source_element_, queue_pre_, NULL);
        
        // Connect pad-added signal for dynamic linking
        g_signal_connect(source_element_, "pad-added", G_CALLBACK(rtsp_pad_added), this);
        
        // Create empty ghost pad (will be set up when pads are added)
        GstPad* ghost_pad = gst_ghost_pad_new_no_target("src", GST_PAD_SRC);
        if (!gst_element_add_pad(bin_, ghost_pad)) {
            std::cerr << "Failed to add ghost pad to source bin" << std::endl;
            return false;
        }
        
        std::cout << "[RTSP] RTSP source created successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[RTSP] Exception creating RTSP source: " << e.what() << std::endl;
        // Fall back to uridecodebin
        return create_uri_decode_source();
    }
}

bool SourceBin::create_uri_decode_source() {
    std::cout << "Creating URI decode source for " << uri_ << std::endl;
    
    // Create uridecodebin
    std::string decode_name = "uri-decode-bin-" + std::to_string(index_);
    source_element_ = gst_element_factory_make("uridecodebin", decode_name.c_str());
    if (!source_element_) {
        std::cerr << "Unable to create uri decode bin" << std::endl;
        return false;
    }
    
    // Set URI
    g_object_set(source_element_, "uri", uri_.c_str(), NULL);
    
    // Setup capabilities if needed
    auto force_nvmm = EnvUtils::env_bool("DS_FORCE_NVMM", false);
    if (force_nvmm.value_or(false)) {
        std::cout << "Forcing NVMM caps for source index=" << index_ << std::endl;
        GstCaps* caps = gst_caps_from_string("video/x-raw(memory:NVMM)");
        g_object_set(source_element_, "caps", caps, NULL);
        gst_caps_unref(caps);
    }
    
    // Disable expose-all-streams if available
    GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(source_element_), "expose-all-streams");
    if (spec) {
        g_object_set(source_element_, "expose-all-streams", FALSE, NULL);
    }
    
    // Connect callbacks
    g_signal_connect(source_element_, "pad-added", G_CALLBACK(cb_newpad), this);
    g_signal_connect(source_element_, "child-added", G_CALLBACK(decodebin_child_added), this);
    
    // Add to bin
    if (!gst_bin_add(GST_BIN(bin_), source_element_)) {
        std::cerr << "Failed to add uridecodebin to source bin" << std::endl;
        return false;
    }
    
    // Create ghost pad
    GstPad* ghost_pad = gst_ghost_pad_new_no_target("src", GST_PAD_SRC);
    if (!gst_element_add_pad(bin_, ghost_pad)) {
        std::cerr << "Failed to add ghost pad to source bin" << std::endl;
        return false;
    }
    
    std::cout << "URI decode source created successfully" << std::endl;
    return true;
}

bool SourceBin::setup_rtsp_properties(GstElement* rtspsrc) {
    if (!rtspsrc) return false;
    
    try {
        // Core RTSP properties
        g_object_set(rtspsrc,
                     "do-rtcp", rtsp_config_.do_rtcp,
                     "do-retransmission", rtsp_config_.do_retransmission,
                     "ntp-sync", rtsp_config_.ntp_sync,
                     "user-agent", rtsp_config_.user_agent.c_str(),
                     NULL);
        
        // Set protocol (TCP/UDP)
        if (rtsp_config_.tcp_mode) {
            GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(rtspsrc), "protocols");
            if (spec) {
                g_object_set(rtspsrc, "protocols", 4, NULL);  // TCP
            }
        }
        
        // Buffer mode
        GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(rtspsrc), "buffer-mode");
        if (spec) {
            g_object_set(rtspsrc, "buffer-mode", rtsp_config_.buffer_mode, NULL);
        }
        
        // Drop on latency
        spec = g_object_class_find_property(G_OBJECT_GET_CLASS(rtspsrc), "drop-on-latency");
        if (spec) {
            g_object_set(rtspsrc, "drop-on-latency", rtsp_config_.drop_on_latency, NULL);
        }
        
        // Latency
        spec = g_object_class_find_property(G_OBJECT_GET_CLASS(rtspsrc), "latency");
        if (spec) {
            g_object_set(rtspsrc, "latency", rtsp_config_.latency, NULL);
        }
        
        // Timeouts
        spec = g_object_class_find_property(G_OBJECT_GET_CLASS(rtspsrc), "timeout");
        if (spec) {
            g_object_set(rtspsrc, "timeout", rtsp_config_.timeout_us, NULL);
        }
        
        spec = g_object_class_find_property(G_OBJECT_GET_CLASS(rtspsrc), "tcp-timeout");
        if (spec) {
            g_object_set(rtspsrc, "tcp-timeout", rtsp_config_.tcp_timeout_us, NULL);
        }
        
        // Retry count
        spec = g_object_class_find_property(G_OBJECT_GET_CLASS(rtspsrc), "retry");
        if (spec) {
            g_object_set(rtspsrc, "retry", rtsp_config_.retry_count, NULL);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception setting RTSP properties: " << e.what() << std::endl;
        return false;
    }
}

void SourceBin::setup_rtsp_queues() {
    // Motion-optimized defaults for better real-time performance
    int q_leaky = EnvUtils::env_int("DS_RTSP_QUEUE_LEAKY", 2);
    int q_max_buffers = EnvUtils::env_int("DS_RTSP_QUEUE_MAX_BUFFERS", 2);
    int q_max_bytes = EnvUtils::env_int("DS_RTSP_QUEUE_MAX_BYTES", 0);
    int q_max_time = EnvUtils::env_int("DS_RTSP_QUEUE_MAX_TIME", 0);
    bool q_silent = EnvUtils::env_bool("DS_RTSP_QUEUE_SILENT", true).value_or(true);
    bool q_flush = EnvUtils::env_bool("DS_RTSP_QUEUE_FLUSH_ON_EOS", true).value_or(true);
    
    // Pre-decode queue
    std::string queue_pre_name = "queue-pre-" + std::to_string(index_);
    queue_pre_ = gst_element_factory_make("queue", queue_pre_name.c_str());
    if (queue_pre_) {
        int pre_max_buffers = EnvUtils::env_int("DS_RTSP_QUEUE_PRE_MAX_BUFFERS", q_max_buffers);
        g_object_set(queue_pre_,
                     "leaky", q_leaky,
                     "max-size-buffers", pre_max_buffers,
                     "max-size-bytes", q_max_bytes,
                     "max-size-time", (guint64)q_max_time,
                     "silent", q_silent,
                     "flush-on-eos", q_flush,
                     NULL);
    }
    
    // Post-decode queue
    std::string queue_post_name = "queue-postdec-" + std::to_string(index_);
    queue_post_ = gst_element_factory_make("queue", queue_post_name.c_str());
    if (queue_post_) {
        int post_max_buffers = EnvUtils::env_int("DS_RTSP_QUEUE_POST_MAX_BUFFERS", 1);
        g_object_set(queue_post_,
                     "leaky", q_leaky,
                     "max-size-buffers", post_max_buffers,
                     "max-size-bytes", q_max_bytes,
                     "max-size-time", (guint64)q_max_time,
                     "silent", q_silent,
                     "flush-on-eos", q_flush,
                     NULL);
    }
}

void SourceBin::cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data) {
    SourceBin* source_bin = static_cast<SourceBin*>(data);
    
    std::cout << "In cb_newpad" << std::endl;
    
    GstCaps* caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps) {
        caps = gst_pad_query_caps(decoder_src_pad, NULL);
    }
    
    if (caps) {
        GstStructure* gststruct = gst_caps_get_structure(caps, 0);
        const gchar* gstname = gst_structure_get_name(gststruct);
        
        std::cout << "gstname=" << gstname << std::endl;
        
        // Check for video stream
        if (g_strrstr(gstname, "video")) {
            GstCapsFeatures* features = gst_caps_get_features(caps, 0);
            std::cout << "Caps features: " << gst_caps_features_to_string(features) << std::endl;
            
            // Check for NVMM memory
            if (gst_caps_features_contains(features, "memory:NVMM")) {
                // Get ghost pad from bin
                GstPad* bin_ghost_pad = gst_element_get_static_pad(GST_ELEMENT(source_bin->bin_), "src");
                if (bin_ghost_pad && GST_IS_GHOST_PAD(bin_ghost_pad)) {
                    if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad), decoder_src_pad)) {
                        std::cerr << "Failed to link decoder src pad to source bin ghost pad" << std::endl;
                    }
                    gst_object_unref(bin_ghost_pad);
                }
            } else {
                std::cerr << "Error: Decodebin did not pick nvidia decoder plugin." << std::endl;
            }
        } else if (g_strrstr(gstname, "audio")) {
            // Consume audio with fakesink
            try {
                GstElement* fake_sink = gst_element_factory_make("fakesink", NULL);
                if (fake_sink) {
                    g_object_set(fake_sink,
                                 "enable-last-sample", FALSE,
                                 "sync", FALSE,
                                 NULL);
                    
                    if (gst_bin_add(GST_BIN(source_bin->bin_), fake_sink)) {
                        gst_element_sync_state_with_parent(fake_sink);
                        GstPad* sinkpad = gst_element_get_static_pad(fake_sink, "sink");
                        if (sinkpad) {
                            if (gst_pad_can_link(decoder_src_pad, sinkpad)) {
                                gst_pad_link(decoder_src_pad, sinkpad);
                            } else {
                                gst_bin_remove(GST_BIN(source_bin->bin_), fake_sink);
                            }
                            gst_object_unref(sinkpad);
                        }
                    }
                }
            } catch (...) {
                // Ignore audio sink creation errors
            }
        }
        
        gst_caps_unref(caps);
    }
}

void SourceBin::decodebin_child_added(GstChildProxy* child_proxy, GObject* object, 
                                     const gchar* name, gpointer user_data) {
    SourceBin* source_bin = static_cast<SourceBin*>(user_data);
    
    std::cout << "Decodebin child added: " << name << std::endl;
    
    if (g_strrstr(name, "decodebin")) {
        g_signal_connect(object, "child-added", G_CALLBACK(decodebin_child_added), user_data);
    }
    
    // Configure RTSP source properties
    if (g_strrstr(name, "source")) {
        try {
            GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "latency");
            if (spec) {
                g_object_set(object, "latency", source_bin->rtsp_config_.latency, NULL);
            }
            
            if (source_bin->rtsp_config_.tcp_mode) {
                spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "protocols");
                if (spec) {
                    g_object_set(object, "protocols", 4, NULL);  // TCP
                }
            }
            
            spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "do-retransmission");
            if (spec) {
                g_object_set(object, "do-retransmission", source_bin->rtsp_config_.do_retransmission, NULL);
            }
            
            spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "drop-on-latency");
            if (spec) {
                g_object_set(object, "drop-on-latency", source_bin->rtsp_config_.drop_on_latency, NULL);
            }
            
            spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "ntp-sync");
            if (spec) {
                g_object_set(object, "ntp-sync", FALSE, NULL);
            }
        } catch (...) {
            // Ignore property setting errors
        }
    }
    
    // Configure NVIDIA decoder for realtime dropping
    extern bool REALTIME_DROP;  // Defined in main.cpp
    
    if (REALTIME_DROP && (g_strrstr(name, "nvv4l2decoder") || 
                         g_strrstr(name, "nvh264dec") || 
                         g_strrstr(name, "nvh265dec"))) {
        try {
            GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "drop-frame-interval");
            if (spec) {
                int dfi = EnvUtils::env_int("DS_DEC_DROP_FRAME_INTERVAL", 1);
                g_object_set(object, "drop-frame-interval", dfi, NULL);
            }
            
            spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "disable-dpb");
            if (spec) {
                bool disable_dpb = EnvUtils::env_bool("DS_DEC_DISABLE_DPB", true).value_or(true);
                g_object_set(object, "disable-dpb", disable_dpb, NULL);
            }
            
            spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "output-io-mode");
            if (spec) {
                int io_mode = EnvUtils::env_int("DS_DEC_OUTPUT_IO_MODE", 2);  // DMABUF
                g_object_set(object, "output-io-mode", io_mode, NULL);
            }
            
            spec = g_object_class_find_property(G_OBJECT_GET_CLASS(object), "max-pool-size");
            if (spec) {
                int pool_size = EnvUtils::env_int("DS_DEC_MAX_POOL_SIZE", 4);
                g_object_set(object, "max-pool-size", pool_size, NULL);
            }
        } catch (...) {
            // Ignore decoder configuration errors
        }
    }
}

void SourceBin::rtsp_pad_added(GstElement* src, GstPad* pad, gpointer user_data) {
    SourceBin* source_bin = static_cast<SourceBin*>(user_data);
    
    if (source_bin->pipeline_linked_) {
        return;  // Already linked
    }
    
    try {
        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (!caps) {
            caps = gst_pad_query_caps(pad, NULL);
        }
        
        if (!caps) {
            std::cerr << "[RTSP] No caps available for pad" << std::endl;
            return;
        }
        
        gchar* caps_str = gst_caps_to_string(caps);
        std::string caps_string = caps_str ? caps_str : "";
        g_free(caps_str);
        
        std::cout << "[RTSP] Pad added with caps: " << caps_string << std::endl;
        
        // Detect codec type
        bool is_h264 = caps_string.find("application/x-rtp") != std::string::npos && 
                       caps_string.find("H264") != std::string::npos;
        bool is_h265 = caps_string.find("application/x-rtp") != std::string::npos && 
                       (caps_string.find("H265") != std::string::npos || caps_string.find("HEVC") != std::string::npos);
        
        if (!is_h264 && !is_h265) {
            std::cout << "[RTSP] Unsupported codec in caps: " << caps_string << std::endl;
            gst_caps_unref(caps);
            return;
        }
        
        // Create appropriate elements based on codec
        GstElement* depay = nullptr;
        GstElement* parse = nullptr;
        GstElement* decoder = nullptr;
        
        if (is_h264) {
            std::cout << "[RTSP] Detected H.264 stream for source " << source_bin->index_ << std::endl;
            depay = gst_element_factory_make("rtph264depay", ("depay-" + std::to_string(source_bin->index_)).c_str());
            parse = gst_element_factory_make("h264parse", ("h264parse-" + std::to_string(source_bin->index_)).c_str());
            
            // Try hardware decoders first
            const char* decoder_candidates[] = {"nvv4l2decoder", "nvh264dec", "avdec_h264", nullptr};
            for (int i = 0; decoder_candidates[i] && !decoder; i++) {
                decoder = gst_element_factory_make(decoder_candidates[i], 
                                                 ("dec-" + std::string(decoder_candidates[i]) + "-" + std::to_string(source_bin->index_)).c_str());
                if (decoder) {
                    std::cout << "[RTSP] Using " << decoder_candidates[i] << " for H.264 decoding" << std::endl;
                }
            }
        } else if (is_h265) {
            std::cout << "[RTSP] Detected H.265/HEVC stream for source " << source_bin->index_ << std::endl;
            depay = gst_element_factory_make("rtph265depay", ("depay-" + std::to_string(source_bin->index_)).c_str());
            parse = gst_element_factory_make("h265parse", ("h265parse-" + std::to_string(source_bin->index_)).c_str());
            
            // Try hardware decoders first
            const char* decoder_candidates[] = {"nvv4l2decoder", "nvh265dec", "avdec_h265", nullptr};
            for (int i = 0; decoder_candidates[i] && !decoder; i++) {
                decoder = gst_element_factory_make(decoder_candidates[i], 
                                                 ("dec-" + std::string(decoder_candidates[i]) + "-" + std::to_string(source_bin->index_)).c_str());
                if (decoder) {
                    std::cout << "[RTSP] Using " << decoder_candidates[i] << " for H.265 decoding" << std::endl;
                }
            }
        }
        
        if (!depay || !parse || !decoder) {
            std::cerr << "[RTSP] Failed to create decode chain elements" << std::endl;
            if (depay) gst_object_unref(depay);
            if (parse) gst_object_unref(parse);
            if (decoder) gst_object_unref(decoder);
            gst_caps_unref(caps);
            return;
        }
        
        // Store dynamic elements
        source_bin->dynamic_elements_.depay = depay;
        source_bin->dynamic_elements_.parse = parse;
        source_bin->dynamic_elements_.decoder = decoder;
        
        // Add elements to bin
        gst_bin_add_many(GST_BIN(source_bin->bin_), depay, parse, decoder, source_bin->queue_post_, NULL);
        
        // Link pad to depay
        GstPad* sinkpad = gst_element_get_static_pad(depay, "sink");
        if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
            std::cerr << "[RTSP] Failed to link rtspsrc to depay" << std::endl;
            gst_object_unref(sinkpad);
            gst_caps_unref(caps);
            return;
        }
        gst_object_unref(sinkpad);
        
        // Link decode chain
        if (!gst_element_link_many(depay, parse, decoder, source_bin->queue_post_, NULL)) {
            std::cerr << "[RTSP] Failed to link decode chain" << std::endl;
            gst_caps_unref(caps);
            return;
        }
        
        // Sync elements with parent state
        gst_element_sync_state_with_parent(depay);
        gst_element_sync_state_with_parent(parse);
        gst_element_sync_state_with_parent(decoder);
        gst_element_sync_state_with_parent(source_bin->queue_post_);
        
        // Set up ghost pad
        GstPad* src_pad = gst_element_get_static_pad(source_bin->queue_post_, "src");
        GstPad* ghost_pad = gst_element_get_static_pad(source_bin->bin_, "src");
        
        if (ghost_pad && GST_IS_GHOST_PAD(ghost_pad) && src_pad) {
            if (gst_ghost_pad_set_target(GST_GHOST_PAD(ghost_pad), src_pad)) {
                source_bin->pipeline_linked_ = true;
                std::cout << "[RTSP] Successfully linked RTSP pipeline for source " << source_bin->index_ << std::endl;
            } else {
                std::cerr << "[RTSP] Failed to set ghost pad target" << std::endl;
            }
        }
        
        if (src_pad) gst_object_unref(src_pad);
        if (ghost_pad) gst_object_unref(ghost_pad);
        gst_caps_unref(caps);
        
    } catch (const std::exception& e) {
        std::cerr << "[RTSP] Exception in pad-added callback: " << e.what() << std::endl;
    }
}

} // namespace EdgeDeepStream