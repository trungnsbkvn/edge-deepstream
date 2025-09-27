#include <gst/gst.h>
#include <iostream>
#include <signal.h>

static GMainLoop *loop = nullptr;
static bool pad_added_called = false;

// Minimal pad-added callback
static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    std::cout << "*** SOURCE BIN PAD-ADDED CALLBACK TRIGGERED! ***" << std::endl;
    pad_added_called = true;
    
    GstCaps *caps = gst_pad_get_current_caps(pad);
    if (caps) {
        gchar *caps_str = gst_caps_to_string(caps);
        std::cout << "New pad caps: " << caps_str << std::endl;
        g_free(caps_str);
        gst_caps_unref(caps);
    }
    
    // Link to ghost pad (this is what our DeepStream code should do)
    GstBin *source_bin = GST_BIN(data);
    GstPad *ghost_pad = gst_element_get_static_pad(GST_ELEMENT(source_bin), "src");
    if (ghost_pad && GST_IS_GHOST_PAD(ghost_pad)) {
        if (gst_ghost_pad_set_target(GST_GHOST_PAD(ghost_pad), pad)) {
            std::cout << "Successfully set ghost pad target" << std::endl;
        } else {
            std::cout << "Failed to set ghost pad target" << std::endl;
        }
        gst_object_unref(ghost_pad);
    }
}

// Bus message handler
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "End of stream" << std::endl;
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            std::cout << "Error: " << error->message << std::endl;
            g_error_free(error);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED: {
            if (GST_MESSAGE_SRC(msg) == GST_OBJECT(data)) {
                GstState old_state, new_state;
                gst_message_parse_state_changed(msg, &old_state, &new_state, nullptr);
                std::cout << "Pipeline state changed from " 
                         << gst_element_state_get_name(old_state) 
                         << " to " << gst_element_state_get_name(new_state) << std::endl;
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

// Signal handler
static void sigint_handler(int signum) {
    std::cout << "Interrupt signal received" << std::endl;
    if (loop) {
        g_main_loop_quit(loop);
    }
}

int main(int argc, char *argv[]) {
    // Initialize GStreamer
    gst_init(&argc, &argv);
    
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <RTSP_URI>" << std::endl;
        std::cout << "Example: " << argv[0] << " rtsp://admin:123456Aa@192.168.0.213:1554/Streaming/Channels/501" << std::endl;
        return -1;
    }
    
    std::string uri = argv[1];
    std::cout << "Testing DeepStream-style source bin with URI: " << uri << std::endl;
    
    // Create pipeline
    GstElement *pipeline = gst_pipeline_new("test-pipeline");
    if (!pipeline) {
        std::cout << "Failed to create pipeline" << std::endl;
        return -1;
    }
    
    // Create source bin (similar to our SourceBin class)
    GstElement *source_bin = gst_bin_new("source-bin");
    if (!source_bin) {
        std::cout << "Failed to create source bin" << std::endl;
        return -1;
    }
    
    // Create uridecodebin
    GstElement *uridecodebin = gst_element_factory_make("uridecodebin", "decoder");
    if (!uridecodebin) {
        std::cout << "Failed to create uridecodebin" << std::endl;
        return -1;
    }
    
    // Set URI
    g_object_set(uridecodebin, "uri", uri.c_str(), NULL);
    
    // Connect pad-added signal
    g_signal_connect(uridecodebin, "pad-added", G_CALLBACK(on_pad_added), source_bin);
    
    // Add uridecodebin to source bin
    gst_bin_add(GST_BIN(source_bin), uridecodebin);
    
    // Create ghost pad (no target initially - will be set by callback)
    GstPad *ghost_pad = gst_ghost_pad_new_no_target("src", GST_PAD_SRC);
    gst_element_add_pad(source_bin, ghost_pad);
    
    // Add source bin to pipeline
    gst_bin_add(GST_BIN(pipeline), source_bin);
    
    // Create fakesink and connect to ghost pad
    GstElement *fakesink = gst_element_factory_make("fakesink", "sink");
    gst_bin_add(GST_BIN(pipeline), fakesink);
    
    // Link source bin ghost pad to fakesink
    GstPad *sink_pad = gst_element_get_static_pad(fakesink, "sink");
    GstPad *src_pad = gst_element_get_static_pad(source_bin, "src");
    gst_pad_link(src_pad, sink_pad);
    gst_object_unref(sink_pad);
    gst_object_unref(src_pad);
    
    // Set up bus
    GstBus *bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, bus_call, pipeline);
    gst_object_unref(bus);
    
    // Set up signal handler
    signal(SIGINT, sigint_handler);
    
    // Create main loop
    loop = g_main_loop_new(NULL, FALSE);
    
    std::cout << "Setting pipeline to PLAYING..." << std::endl;
    
    // Start playing
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cout << "Unable to set pipeline to PLAYING state" << std::endl;
        return -1;
    }
    
    std::cout << "Pipeline started. Waiting for pad-added signal..." << std::endl;
    std::cout << "Press Ctrl+C to quit" << std::endl;
    
    // Run main loop
    g_main_loop_run(loop);
    
    // Cleanup
    std::cout << "Shutting down..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    
    if (pad_added_called) {
        std::cout << "✅ SUCCESS: SOURCE BIN pad-added callback was triggered!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ FAILURE: SOURCE BIN pad-added callback was NOT triggered!" << std::endl;
        return 1;
    }
}