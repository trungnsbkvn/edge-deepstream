#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <iomanip>
#include <map>
#include <vector>
#include <unordered_map>

// DeepStream includes
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

// Config parsing
#include "config_parser.h"
#include "edge_deepstream.h"
#include "probe.h"

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr("Error: %s\n", error->message); \
        g_error_free(error); \
        return -1; \
    }

static GMainLoop *loop = NULL;
static GstElement *pipeline = NULL;
static GstElement *streammux = NULL;
static std::vector<GstElement*> source_bins;
static std::vector<GstElement*> video_converters;
static GstElement *pgie = NULL;
static GstElement *sgie = NULL;
static GstElement *tracker = NULL;
static GstElement *nvosd = NULL;
static GstElement *tiler = NULL;
static GstElement *sink = NULL;

static void source_pad_added_callback(GstElement *src, GstPad *new_pad, GstElement *converter) {
    // Get the sink pad index from the source element
    int sink_pad_index = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(src), "sink_pad_index"));

    g_print("Pad-added signal received for source %d\n", sink_pad_index);

    // Check if pad is video
    GstCaps *caps = gst_pad_get_current_caps(new_pad);
    if (!caps) {
        g_print("No caps on pad\n");
        return;
    }

    const GstStructure *str = gst_caps_get_structure(caps, 0);
    g_print("Pad caps: %s\n", gst_structure_get_name(str));

    if (!g_str_has_prefix(gst_structure_get_name(str), "video/")) {
        gst_caps_unref(caps);
        g_print("Not a video pad, ignoring\n");
        return;
    }

    // Print caps details for debugging
    gchar *caps_str = gst_caps_to_string(caps);
    g_print("Video caps: %s\n", caps_str);
    g_free(caps_str);
    gst_caps_unref(caps);

    // Get sink pad from converter
    GstPad *sink_pad = gst_element_get_static_pad(converter, "sink");

    if (!sink_pad) {
        g_printerr("Failed to get sink pad from converter for source %d\n", sink_pad_index);
        return;
    }

    g_print("Linking video pad to converter sink for source %d\n", sink_pad_index);

    // Check if pads can be linked
    GstPadLinkReturn link_ret = gst_pad_link(new_pad, sink_pad);
    if (link_ret != GST_PAD_LINK_OK) {
        g_printerr("Failed to link source %d to converter: %s\n", sink_pad_index, gst_pad_link_get_name(link_ret));

        // Print pad caps for debugging
        GstCaps *src_caps = gst_pad_get_current_caps(new_pad);
        GstCaps *sink_caps = gst_pad_get_current_caps(sink_pad);

        if (src_caps) {
            gchar *src_caps_str = gst_caps_to_string(src_caps);
            g_printerr("Source pad caps: %s\n", src_caps_str);
            g_free(src_caps_str);
            gst_caps_unref(src_caps);
        }

        if (sink_caps) {
            gchar *sink_caps_str = gst_caps_to_string(sink_caps);
            g_printerr("Sink pad caps: %s\n", sink_caps_str);
            g_free(sink_caps_str);
            gst_caps_unref(sink_caps);
        }
    } else {
        g_print("Successfully linked source %d to converter\n", sink_pad_index);
    }

    gst_object_unref(sink_pad);
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            g_printerr("Debug info: %s\n", debug ? debug : "none");
            g_error_free(error);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED: {
            GstState old_state, new_state;
            gst_message_parse_state_changed(msg, &old_state, &new_state, NULL);
            if (GST_OBJECT_NAME(msg->src) == std::string("pipeline")) {
                g_print("Pipeline state changed from %s to %s\n",
                       gst_element_state_get_name(old_state),
                       gst_element_state_get_name(new_state));
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // Parse config file
    std::string config_path = "/home/m2n/edge-deepstream/config/config_pipeline.toml";
    auto config = EdgeDeepStream::ConfigParser::parse_toml(config_path);
    if (!config) {
        g_printerr("Failed to parse config file: %s\n", config_path.c_str());
        return -1;
    }

    // Extract verbose setting from debug section
    bool verbose = false;
    if (config->has_section("debug")) {
        auto& debug_section = config->sections["debug"];
        if (debug_section.find("verbose") != debug_section.end()) {
            verbose = (debug_section["verbose"] == "1" || debug_section["verbose"] == "true");
        }
    }

    // Create pipeline
    pipeline = gst_pipeline_new("edge-deepstream-pipeline");
    if (!pipeline) {
        g_printerr("Failed to create pipeline\n");
        return -1;
    }

    // Create elements from config
    streammux = gst_element_factory_make("nvstreammux", "streammux");
    if (!streammux) {
        g_printerr("Failed to create streammux\n");
        return -1;
    }

    // Set streammux properties from config
    if (config->has_section("streammux")) {
        EdgeDeepStream::ConfigParser::set_element_properties(streammux, config->sections["streammux"]);
    }

    // Create PGIE
    pgie = gst_element_factory_make("nvinfer", "pgie");
    if (!pgie) {
        g_printerr("Failed to create pgie\n");
        return -1;
    }

    // Set PGIE properties from config
    if (config->has_section("pgie")) {
        EdgeDeepStream::ConfigParser::set_element_properties(pgie, config->sections["pgie"]);
    }

    // Create SGIE
    sgie = gst_element_factory_make("nvinfer", "sgie");
    if (!sgie) {
        g_printerr("Failed to create sgie\n");
        return -1;
    }

    // Set SGIE properties from config
    if (config->has_section("sgie")) {
        EdgeDeepStream::ConfigParser::set_element_properties(sgie, config->sections["sgie"]);
    }

    // Create tracker
    tracker = gst_element_factory_make("nvtracker", "tracker");
    if (!tracker) {
        g_printerr("Failed to create tracker\n");
        return -1;
    } else {
        g_print("Successfully created tracker\n");
    }

    // Set tracker properties from config
    if (config->has_section("tracker")) {
        auto tracker_config = config->sections["tracker"];
        auto config_file_it = tracker_config.find("config-file-path");
        if (config_file_it != tracker_config.end()) {
            EdgeDeepStream::ConfigParser::set_tracker_properties(tracker, config_file_it->second);
            g_print("Tracker properties set from config\n");
        }
    }

    // Get display mode from config
    bool enable_display = config->get<int>("pipeline", "display", 0);

    // Detect display availability; fallback to fakesink on headless/TTY sessions
    const char* display_env = getenv("DISPLAY");
    const char* wayland_env = getenv("WAYLAND_DISPLAY");
    bool has_display = (display_env != nullptr && strlen(display_env) > 0) ||
                      (wayland_env != nullptr && strlen(wayland_env) > 0);
    bool headless = !has_display;

    // Create sink based on display mode and platform
    if (!enable_display || headless) {
        if (headless && enable_display) {
            g_print("No GUI display detected (DISPLAY/WAYLAND_DISPLAY unset). Falling back to fakesink for headless run.\n");
        }
        g_print("Creating Fakesink \n");
        sink = gst_element_factory_make("fakesink", "fakesink");
        if (sink) {
            g_object_set(G_OBJECT(sink), "enable-last-sample", 0, "sync", 0, NULL);
        }
    } else {
        bool is_aarch64 = config->get<int>("pipeline", "is_aarch64", 0);
        if (is_aarch64) {
            g_print("Creating nv3dsink \n");
            sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
            if (!sink) {
                g_printerr("Failed to create nv3dsink \n");
            }
        } else {
            g_print("Creating EGLSink \n");
            sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
            if (!sink) {
                g_printerr("Failed to create egl sink \n");
            }
        }
    }

    if (!sink) {
        g_printerr("Unable to create sink element, using fakesink as fallback\n");
        sink = gst_element_factory_make("fakesink", "fakesink");
        if (sink) {
            g_object_set(G_OBJECT(sink), "enable-last-sample", 0, "sync", 0, NULL);
        }
    }

    if (!sink) {
        g_printerr("Failed to create sink\n");
        return -1;
    }

    // Print which sink was created
    g_print("Created sink: %s\n", GST_ELEMENT_NAME(sink));

    // Set sink properties from config
    if (config->has_section("sink")) {
        EdgeDeepStream::ConfigParser::set_element_properties(sink, config->sections["sink"]);
    }

    // Additional sink properties for better visibility on Jetson
    if (enable_display) {
        g_print("Setting additional sink properties for display visibility...\n");
        // Try to set window properties if available
        g_object_set(G_OBJECT(sink),
                     "force-aspect-ratio", TRUE,
                     NULL);
        g_print("Sink properties set for display\n");
    }

    // Check if display is enabled
    g_print("Display mode: %s\n", enable_display ? "enabled" : "disabled (headless)");

    // Create display elements if needed
    if (enable_display) {        
        // Create tiler
        tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
        if (!tiler) {
            g_printerr("Failed to create tiler\n");
            return -1;
        } else {
            g_print("Successfully created tiler\n");
        }

        // Set tiler properties from config
        if (config->has_section("tiler")) {
            EdgeDeepStream::ConfigParser::set_element_properties(tiler, config->sections["tiler"]);
            g_print("Tiler properties set: width=%d, height=%d\n",
                    config->get<int>("tiler", "width", 1280),
                    config->get<int>("tiler", "height", 720));
        }

        // Create nvosd
        nvosd = gst_element_factory_make("nvdsosd", "nvosd");
        if (!nvosd) {
            g_printerr("Failed to create nvosd\n");
            return -1;
        } else {
            g_print("Successfully created nvosd\n");
        }

        // Set nvosd properties from config
        if (config->has_section("nvosd")) {
            EdgeDeepStream::ConfigParser::set_element_properties(nvosd, config->sections["nvosd"]);
            g_print("NVOSD properties set: process-mode=%d, display-text=%d\n",
                    config->get<int>("nvosd", "process-mode", 0),
                    config->get<int>("nvosd", "display-text", 1));
        }

        g_print("Display elements created (with tracker)\n");
    }

    // Add elements to pipeline
    if (enable_display) {
        g_print("Adding display elements to pipeline...\n");
        gst_bin_add_many(GST_BIN(pipeline), streammux, pgie, sgie, tracker, tiler, nvosd, sink, NULL);

        // Link elements: streammux -> pgie -> tracker -> sgie -> tiler -> nvosd -> sink
        g_print("Linking display pipeline elements...\n");
        if (!gst_element_link_many(streammux, pgie, tracker, sgie, tiler, nvosd, sink, NULL)) {
            g_printerr("Failed to link display pipeline elements\n");
            return -1;
        } else {
            g_print("Successfully linked display pipeline: streammux -> pgie -> tracker -> sgie -> tiler -> nvosd -> sink\n");
        }
    } else {
        gst_bin_add_many(GST_BIN(pipeline), streammux, pgie, sgie, sink, NULL);

        // Link elements: streammux -> pgie -> sgie -> sink
        if (!gst_element_link_many(streammux, pgie, sgie, sink, NULL)) {
            g_printerr("Failed to link headless pipeline elements\n");
            return -1;
        }
    }

    // Create source bins for all sources from config BEFORE setting pipeline to READY
    if (config->has_section("source")) {
        auto& source_config = config->sections["source"];

        // Iterate through all sources in the config
        int source_index = 0;
        for (const auto& source_pair : source_config) {
            std::string source_key = source_pair.first;
            std::string source_uri = source_pair.second;

            if (source_uri.empty()) continue;

            g_print("Creating source '%s': %s\n", source_key.c_str(), source_uri.c_str());

            // Determine source type from URI
            std::string source_type = "unknown";
            if (source_uri.find("file://") == 0) {
                source_type = "file";
            } else if (source_uri.find("rtsp://") == 0) {
                source_type = "rtsp";
            }

            g_print("Detected source type: %s\n", source_type.c_str());

            // Create source element
            GstElement *src_bin = gst_element_factory_make("uridecodebin", source_key.c_str());
            if (!src_bin) {
                g_printerr("Failed to create source for %s\n", source_key.c_str());
                continue;
            }

            // Create video converter for this source
            std::string converter_name = "converter_" + std::to_string(source_index);
            GstElement *converter = gst_element_factory_make("nvvideoconvert", converter_name.c_str());
            if (!converter) {
                g_printerr("Failed to create video converter for %s\n", source_key.c_str());
                gst_object_unref(src_bin);
                continue;
            }

            g_object_set(G_OBJECT(src_bin), "uri", source_uri.c_str(), NULL);
            gst_bin_add_many(GST_BIN(pipeline), src_bin, converter, NULL);
            source_bins.push_back(src_bin);
            video_converters.push_back(converter);

            // Store the sink pad index for this source
            g_object_set_data(G_OBJECT(src_bin), "sink_pad_index", GINT_TO_POINTER(source_index));

            // Connect source to converter
            g_signal_connect(src_bin, "pad-added", G_CALLBACK(source_pad_added_callback), converter);

            // Link converter to streammux
            gchar *sink_pad_name = g_strdup_printf("sink_%d", source_index);
            GstPad *sink_pad = gst_element_get_request_pad(streammux, sink_pad_name);
            g_free(sink_pad_name);

            if (!sink_pad) {
                g_printerr("Failed to get sink pad sink_%d from streammux\n", source_index);
                continue;
            }

            GstPad *converter_src_pad = gst_element_get_static_pad(converter, "src");
            if (!converter_src_pad) {
                g_printerr("Failed to get src pad from converter for source %d\n", source_index);
                gst_object_unref(sink_pad);
                continue;
            }

            if (gst_pad_link(converter_src_pad, sink_pad) != GST_PAD_LINK_OK) {
                g_printerr("Failed to link converter to streammux for source %d\n", source_index);
                gst_object_unref(converter_src_pad);
                gst_object_unref(sink_pad);
                continue;
            }

            gst_object_unref(converter_src_pad);
            gst_object_unref(sink_pad);

            source_index++;
        }

        g_print("Created %d sources\n", (int)source_bins.size());
    }

    // Set pipeline to READY state after adding sources
    g_print("Setting pipeline to READY state...\n");
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_READY);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to set pipeline to READY state\n");
        return -1;
    }
    g_print("Pipeline set to READY state\n");

    // Attach probes to PGIE and SGIE using centralized probe logic
    if (!EdgeDeepStream::attach_probes(pgie, sgie, nullptr, verbose)) {
        g_printerr("Failed to attach probes to pipeline elements\n");
        return -1;
    }

    // Add bus watch
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, NULL);
    gst_object_unref(bus);

    // Set pipeline to playing
    g_print("Starting EdgeDeepStream pipeline test...\n");
    ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to start pipeline\n");
        return -1;
    }
    g_print("Pipeline set to PLAYING state\n");

    // Run for 60 seconds to see detection results
    g_print("Running inference for 60 seconds to verify detections...\n");
    std::this_thread::sleep_for(std::chrono::seconds(60));

    // Stop pipeline
    g_print("Stopping pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);

    // Cleanup
    gst_object_unref(pipeline);
    g_print("Config-driven pipeline test completed successfully!\n");

    return 0;
}