#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <signal.h>

#include "config_parser.h"
#include "env_utils.h"
#include "status_codes.h"

using namespace EdgeDeepStream;

// Simple version for initial testing
static GMainLoop* loop = nullptr;

void signal_handler(int sig) {
    std::cout << "Received signal " << sig << ", shutting down..." << std::endl;
    if (loop) {
        g_main_loop_quit(loop);
    }
}

gboolean bus_callback(GstBus* bus, GstMessage* msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* debug_info = nullptr;
            gst_message_parse_error(msg, &err, &debug_info);
            std::cerr << "Error: " << err->message << std::endl;
            if (debug_info) {
                std::cerr << "Debug: " << debug_info << std::endl;
                g_free(debug_info);
            }
            g_error_free(err);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_EOS:
            std::cout << "End-Of-Stream reached" << std::endl;
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_STATE_CHANGED: {
            GstState old_state, new_state, pending_state;
            gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
            std::cout << "State changed: " << gst_element_state_get_name(old_state) 
                     << " -> " << gst_element_state_get_name(new_state) << std::endl;
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char* argv[]) {
    // Initialize GStreamer
    gst_init(&argc, &argv);
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return -1;
    }
    
    std::string config_path = argv[1];
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "EdgeDeepStream C++ Simple Version" << std::endl;
    std::cout << "Config: " << config_path << std::endl;
    
    // Parse configuration
    auto config = ConfigParser::parse_toml(config_path);
    if (!config) {
        std::cerr << "Failed to parse configuration" << std::endl;
        return -1;
    }
    
    std::cout << "Configuration loaded successfully" << std::endl;
    
    // Test environment variables
    auto realtime = EnvUtils::env_bool("DS_REALTIME_DROP");
    std::cout << "DS_REALTIME_DROP: " << (realtime.has_value() ? (realtime.value() ? "true" : "false") : "not set") << std::endl;
    
    // Test status codes
    std::cout << "Status code " << StatusCodes::STATUS_OK_GENERIC << ": " 
              << StatusCodes::get_status_description(StatusCodes::STATUS_OK_GENERIC) << std::endl;
    
    // Create a simple test pipeline
    std::cout << "Creating test pipeline..." << std::endl;
    
    GstElement* pipeline = gst_pipeline_new("test-pipeline");
    GstElement* videotestsrc = gst_element_factory_make("videotestsrc", "source");
    GstElement* sink = gst_element_factory_make("fakesink", "sink");
    
    if (!pipeline || !videotestsrc || !sink) {
        std::cerr << "Failed to create pipeline elements" << std::endl;
        return -1;
    }
    
    // Configure elements
    g_object_set(videotestsrc, "num-buffers", 100, NULL);
    g_object_set(sink, "sync", FALSE, NULL);
    
    // Build pipeline
    gst_bin_add_many(GST_BIN(pipeline), videotestsrc, sink, NULL);
    
    if (!gst_element_link(videotestsrc, sink)) {
        std::cerr << "Failed to link pipeline elements" << std::endl;
        gst_object_unref(pipeline);
        return -1;
    }
    
    // Set up bus callback
    GstBus* bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, bus_callback, nullptr);
    gst_object_unref(bus);
    
    // Start pipeline
    std::cout << "Starting pipeline..." << std::endl;
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Unable to set pipeline to playing state" << std::endl;
        gst_object_unref(pipeline);
        return -1;
    }
    
    // Create main loop and run
    loop = g_main_loop_new(nullptr, FALSE);
    std::cout << "Running pipeline..." << std::endl;
    g_main_loop_run(loop);
    
    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    
    std::cout << "Done!" << std::endl;
    return 0;
}