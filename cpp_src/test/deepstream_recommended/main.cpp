#include "../utils/config_parser.h"
#include "source_bin.h"

// Minimal DeepStream pipeline: read config from TOML, load YOLOv8n PGIE, ArcFace SGIE, and display. Simple logic for validation.
#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <exception>
#include <toml.hpp>

static void pad_added_handler(GstElement *src, GstPad *new_pad, gpointer data) {
    GstElement *queue = (GstElement *)data;
    GstPad *sink_pad = gst_element_get_static_pad(queue, "sink");
    if (!gst_pad_is_linked(sink_pad)) {
        gst_pad_link(new_pad, sink_pad);
    }
    gst_object_unref(sink_pad);
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    std::string config_path = "/home/m2n/edge-deepstream/config/config_pipeline.toml";
    if (argc > 1) config_path = argv[1];
    
    // Parse TOML config
    toml::value config;
    try {
        config = toml::parse(config_path);
        std::cout << "Successfully parsed config: " << config_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed to parse config: " << e.what() << std::endl;
        return -1;
    }
    
    // Extract pipeline config
    bool display_enabled = toml::find_or<bool>(config, "pipeline", "display", true);
    bool is_aarch64 = toml::find_or<bool>(config, "pipeline", "is_aarch64", false);
    bool realtime = toml::find_or<bool>(config, "pipeline", "realtime", true);
    std::string known_face_dir = toml::find_or<std::string>(config, "pipeline", "known_face_dir", "data/known_faces");
    std::string save_feature_path = toml::find_or<std::string>(config, "pipeline", "save_feature_path", "data/features");
    bool save_feature = toml::find_or<bool>(config, "pipeline", "save_feature", false);
    
    // Get camera URIs from [source] table
    std::vector<std::string> camera_uris;
    if (config.contains("source")) {
        const auto& src_table = toml::find(config, "source");
        for (const auto& kv : src_table.as_table()) {
            camera_uris.push_back(kv.second.as_string());
        }
    }
    if (camera_uris.empty()) {
        std::cout << "No camera URIs found in config." << std::endl;
        return -1;
    }
    
    // Get streammux params
    auto streammux_cfg = toml::find(config, "streammux");
    int batch_size = camera_uris.size();
    int width = toml::find_or<int>(streammux_cfg, "width", 960);
    int height = toml::find_or<int>(streammux_cfg, "height", 540);
    int gpu_id = toml::find_or<int>(streammux_cfg, "gpu_id", 0);
    int batched_push_timeout = toml::find_or<int>(streammux_cfg, "batched-push-timeout", 15000);
    int enable_padding = toml::find_or<int>(streammux_cfg, "enable-padding", 1);
    int nvbuf_memory_type = toml::find_or<int>(streammux_cfg, "nvbuf-memory-type", 0);
    int live_source = toml::find_or<int>(streammux_cfg, "live-source", 1);
    int sync_inputs = toml::find_or<int>(streammux_cfg, "sync-inputs", 0);
    int attach_sys_ts = toml::find_or<int>(streammux_cfg, "attach-sys-ts", 1);
    
    // Get inference config paths
    std::string pgie_config = toml::find_or<std::string>(toml::find(config, "pgie"), "config-file-path", "");
    std::string sgie_config = toml::find_or<std::string>(toml::find(config, "sgie"), "config-file-path", "");
    
    // Get tracker config
    std::string tracker_config = toml::find_or<std::string>(toml::find(config, "tracker"), "config-file-path", "");
    
    // Get tiler params
    auto tiler_cfg = toml::find(config, "tiler");
    int tiler_width = toml::find_or<int>(tiler_cfg, "width", 1280);
    int tiler_height = toml::find_or<int>(tiler_cfg, "height", 720);
    
    // Get OSD params
    auto nvosd_cfg = toml::find(config, "nvosd");
    int display_text = toml::find_or<int>(nvosd_cfg, "display-text", 1);
    int process_mode = toml::find_or<int>(nvosd_cfg, "process-mode", 0);
    
    // Get sink params
    auto sink_cfg = toml::find(config, "sink");
    int sink_sync = toml::find_or<int>(sink_cfg, "sync", 0);
    int sink_qos = toml::find_or<int>(sink_cfg, "qos", 1);
    
    std::cout << "Config loaded:" << std::endl;
    std::cout << "  Display: " << (display_enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "  AArch64: " << (is_aarch64 ? "yes" : "no") << std::endl;
    std::cout << "  Realtime: " << (realtime ? "yes" : "no") << std::endl;
    std::cout << "  Cameras: " << camera_uris.size() << std::endl;
    std::cout << "  PGIE config: " << pgie_config << std::endl;
    std::cout << "  SGIE config: " << sgie_config << std::endl;
    std::cout << "  Tracker config: " << tracker_config << std::endl;

    // Create pipeline elements
    GstElement *pipeline = gst_pipeline_new("ds-pipeline");
    GstElement *streammux = gst_element_factory_make("nvstreammux", "streammux");
    g_object_set(streammux,
        "batch-size", batch_size,
        "width", width,
        "height", height,
        "gpu-id", gpu_id,
        "batched-push-timeout", batched_push_timeout,
        "enable-padding", enable_padding,
        "nvbuf-memory-type", nvbuf_memory_type,
        "live-source", live_source,
        "sync-inputs", sync_inputs,
        "attach-sys-ts", attach_sys_ts,
        NULL);
    gst_bin_add(GST_BIN(pipeline), streammux);
    
    // Create source bins
    std::vector<std::unique_ptr<EdgeDeepStream::SourceBin>> source_bins;
    for (size_t i = 0; i < camera_uris.size(); ++i) {
        auto src_bin = std::make_unique<EdgeDeepStream::SourceBin>(i, camera_uris[i]);
        if (!src_bin->create()) {
            std::cerr << "Failed to create SourceBin for camera " << i << std::endl;
            continue;
        }
        gst_bin_add(GST_BIN(pipeline), src_bin->get_bin());
        std::string sinkpad_name = "sink_" + std::to_string(i);
        GstPad *sinkpad = gst_element_get_request_pad(streammux, sinkpad_name.c_str());
        GstPad *srcpad = gst_element_get_static_pad(src_bin->get_bin(), "src");
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            std::cerr << "Failed to link SourceBin " << i << " to streammux" << std::endl;
        } else {
            std::cout << "Linked SourceBin " << i << " to streammux" << std::endl;
        }
        gst_object_unref(sinkpad);
        gst_object_unref(srcpad);
        source_bins.push_back(std::move(src_bin));
    }
    
    // Create pipeline elements with proper config
    GstElement *queue1 = gst_element_factory_make("queue", "queue1");
    GstElement *queue2 = gst_element_factory_make("queue", "queue2");
    GstElement *queue3 = gst_element_factory_make("queue", "queue3");
    GstElement *queue4 = gst_element_factory_make("queue", "queue4");
    GstElement *queue5 = gst_element_factory_make("queue", "queue5");
    GstElement *queue6 = gst_element_factory_make("queue", "queue6");
    GstElement *queue7 = gst_element_factory_make("queue", "queue7");
    
    // Setup queue properties for real-time performance
    int q_leaky = 2, q_max_buf = 3, q_max_bytes = 0, q_max_time = 0;
    bool q_silent = true;
    GstElement* queues[] = {queue1, queue2, queue3, queue4, queue5, queue6, queue7};
    for (auto q : queues) {
        g_object_set(q, "leaky", q_leaky, "max-size-buffers", q_max_buf, 
                    "max-size-bytes", q_max_bytes, "max-size-time", q_max_time, 
                    "silent", q_silent, NULL);
    }
    
    // Create inference elements
    GstElement *pgie = gst_element_factory_make("nvinfer", "primary-inference");
    g_object_set(pgie, "config-file-path", pgie_config.c_str(), NULL);
    
    GstElement *tracker = gst_element_factory_make("nvtracker", "tracker");
    if (!tracker_config.empty()) {
        EdgeDeepStream::ConfigParser::set_tracker_properties(tracker, tracker_config);
    }
    
    GstElement *sgie = gst_element_factory_make("nvinfer", "secondary-inference");
    g_object_set(sgie, "config-file-path", sgie_config.c_str(), NULL);
    
    // Create remaining elements
    GstElement *tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
    g_object_set(tiler, "width", tiler_width, "height", tiler_height, NULL);
    
    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    GstElement *osd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    g_object_set(osd, "display-text", display_text, "process-mode", process_mode, NULL);
    
    // Create sink based on platform and display settings
    GstElement *sink = nullptr;
    if (!display_enabled) {
        sink = gst_element_factory_make("fakesink", "fakesink");
        g_object_set(sink, "enable-last-sample", 0, "sync", 0, NULL);
    } else if (is_aarch64) {
        sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
    } else {
        sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
    }
    g_object_set(sink, "sync", sink_sync, "qos", sink_qos, NULL);
    gst_bin_add_many(GST_BIN(pipeline), queue1, queue2, queue3, queue4, queue5, queue6, queue7, pgie, tracker, sgie, tiler, nvvidconv, osd, sink, NULL);
    // Link elements in recommended order
    if (!gst_element_link_many(streammux, queue1, pgie, queue2, tracker, queue3, sgie, queue4, tiler, queue5, nvvidconv, queue6, osd, queue7, sink, NULL)) {
        std::cerr << "Failed to link pipeline elements" << std::endl;
    gst_object_unref(pipeline);
    // Only unref loop if it was created
    // g_main_loop_unref(loop); // Remove or guard this if loop is not yet created
    return -1;
    }

    // Add bus callback for robust error/warning/info handling
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    struct BusContext {
        GMainLoop *loop;
        GstElement *pipeline;
    } bus_ctx{loop, pipeline};
    auto bus_func = [](GstBus *bus, GstMessage *msg, gpointer data) -> gboolean {
        BusContext *ctx = static_cast<BusContext*>(data);
        GMainLoop *loop = ctx->loop;
        GstElement *pipeline = ctx->pipeline;
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError *err = nullptr; gchar *dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                std::cerr << "[BUS] ERROR from " << GST_OBJECT_NAME(msg->src) << ": " << err->message << std::endl;
                if (dbg) std::cerr << "[BUS] Debug: " << dbg << std::endl;
                g_error_free(err); g_free(dbg);
                if (loop) g_main_loop_quit(loop);
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError *err = nullptr; gchar *dbg = nullptr;
                gst_message_parse_warning(msg, &err, &dbg);
                std::cerr << "[BUS] WARNING from " << GST_OBJECT_NAME(msg->src) << ": " << err->message << std::endl;
                if (dbg) std::cerr << "[BUS] Debug: " << dbg << std::endl;
                g_error_free(err); g_free(dbg);
                break;
            }
            case GST_MESSAGE_INFO: {
                GError *err = nullptr; gchar *dbg = nullptr;
                gst_message_parse_info(msg, &err, &dbg);
                std::cout << "[BUS] INFO from " << GST_OBJECT_NAME(msg->src) << ": " << err->message << std::endl;
                if (dbg) std::cout << "[BUS] Debug: " << dbg << std::endl;
                g_error_free(err); g_free(dbg);
                break;
            }
            case GST_MESSAGE_STATE_CHANGED: {
                if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline)) {
                    GstState oldstate, newstate, pending;
                    gst_message_parse_state_changed(msg, &oldstate, &newstate, &pending);
                    std::cout << "[BUS] Pipeline state changed from " << gst_element_state_get_name(oldstate)
                              << " to " << gst_element_state_get_name(newstate) << std::endl;
                }
                break;
            }
            default: break;
        }
        return TRUE;
    };
    gst_bus_add_watch(bus, bus_func, &bus_ctx);
    gst_object_unref(bus);

    // Add pad probe to streammux src for buffer health
    struct Health { guint64 mux_buffers=0; bool warned=false; } health;
    GstPad *mux_srcpad = gst_element_get_static_pad(streammux, "src");
    if (mux_srcpad) {
        gst_pad_add_probe(mux_srcpad, GST_PAD_PROBE_TYPE_BUFFER,
            [](GstPad*, GstPadProbeInfo*, gpointer data) -> GstPadProbeReturn {
                auto *h = reinterpret_cast<Health*>(data); h->mux_buffers++;
                return GST_PAD_PROBE_OK;
            }, &health, nullptr);
        gst_object_unref(mux_srcpad);
    }

    // Periodic health check (3s)
    g_timeout_add(3000, [](gpointer d) -> gboolean {
        Health *h = reinterpret_cast<Health*>(d);
        std::cout << "[HEALTH] mux_buffers=" << h->mux_buffers << std::endl;
        if (h->mux_buffers == 0 && !h->warned) {
            static int cycles = 0;
            if (++cycles >= 3) {
                std::cout << "[WARN] No buffers yet from streammux" << std::endl;
                h->warned = true;
            }
        }
        return TRUE;
    }, &health);

    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cout << "Failed to set pipeline to PLAYING." << std::endl;
        gst_object_unref(pipeline);
        g_main_loop_unref(loop);
        return -1;
    }
    std::cout << "Pipeline running. Press Ctrl+C to exit." << std::endl;
    g_main_loop_run(loop);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    return 0;
}
