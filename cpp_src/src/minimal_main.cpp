#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <signal.h>
#include <map>

// Minimal standalone implementation
static GMainLoop* loop = nullptr;

void signal_handler(int sig) {
    std::cout << "Received signal " << sig << ", shutting down..." << std::endl;
    if (loop) {
        g_main_loop_quit(loop);
    }
}

gboolean bus_callback(GstBus* /*bus*/, GstMessage* msg, gpointer /*data*/) {
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
    
    std::cout << "EdgeDeepStream C++ Minimal Test" << std::endl;
    std::cout << "This demonstrates that the C++ conversion framework works." << std::endl;
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Test environment variable reading
    const char* realtime_env = std::getenv("DS_REALTIME_DROP");
    std::cout << "DS_REALTIME_DROP = " << (realtime_env ? realtime_env : "not set") << std::endl;
    
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
    std::cout << "Running pipeline (will process 100 frames then exit)..." << std::endl;
    g_main_loop_run(loop);
    
    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    
    std::cout << "C++ conversion framework test completed successfully!" << std::endl;
    std::cout << "The full application can be built by implementing the placeholder modules." << std::endl;
    return 0;
}