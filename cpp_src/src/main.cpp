#include "edge_deepstream.h"
#include "pipeline.h"  // Include full definition after edge_deepstream.h
#include "env_utils.h"
#include "config_parser.h"
#include "event_sender.h"
#include "mqtt_listener.h"
#include "tensorrt_infer.h"

#include <iostream>
#include <signal.h>
#include <algorithm>
#include <cctype>

using namespace EdgeDeepStream;

// Global application instance for signal handling
static Application* g_app = nullptr;

// Signal handler
void signal_handler(int sig) {
    std::cout << "Received signal " << sig << ", shutting down..." << std::endl;
    if (g_app) {
        g_app->shutdown();
    }
}

int main(int argc, char* argv[]) {
    // Initialize GStreamer
    gst_init(&argc, &argv);
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file> [duration_ms]" << std::endl;
        return -1;
    }
    
    std::string config_path = argv[1];
    int duration_ms = -1;  // Run indefinitely by default
    
    if (argc >= 3) {
        try {
            duration_ms = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid duration: " << argv[2] << std::endl;
            return -1;
        }
    }
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create and initialize application
    Application app;
    g_app = &app;
    
    std::cout << "EdgeDeepStream C++ Version" << std::endl;
    std::cout << "Config: " << config_path << std::endl;
    
    if (!app.initialize(config_path)) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }
    
    std::cout << "Application initialized successfully" << std::endl;
    
    // Setup GStreamer plugins
    if (!setup_gstreamer_plugins()) {
        std::cerr << "Failed to setup GStreamer plugins" << std::endl;
        return -1;
    }
    
    // Run the application
    bool success = app.run(duration_ms);
    
    std::cout << "Application finished" << std::endl;
    
    g_app = nullptr;
    return success ? 0 : -1;
}

namespace EdgeDeepStream {

// Global realtime flag
bool REALTIME_DROP = true;

// Application implementation
Application::Application() 
    : running_(false), loop_(nullptr) {
}

Application::~Application() {
    shutdown();
}

bool Application::initialize(const std::string& config_path) {
    try {
        // Parse configuration
        auto config = ConfigParser::parse_toml(config_path);
        if (!config) {
            std::cerr << "Failed to parse configuration file: " << config_path << std::endl;
            return false;
        }
        config_ = *config;
        
        // Set realtime drop policy early
        auto realtime_env = EnvUtils::env_bool("DS_REALTIME_DROP");
        if (realtime_env.has_value()) {
            REALTIME_DROP = realtime_env.value();
        } else {
            // Check config file
            int realtime_config = 0;
            try {
                auto realtime_str = config_.get<std::string>("pipeline", "realtime", "0");
                realtime_config = std::stoi(realtime_str);
            } catch (...) {
                realtime_config = 0;
            }
            REALTIME_DROP = (realtime_config != 0);
        }
        
        std::cout << "REALTIME_DROP = " << (REALTIME_DROP ? "true" : "false") << std::endl;
        
        // Initialize GLib main loop
        loop_ = g_main_loop_new(nullptr, FALSE);
        if (!loop_) {
            std::cerr << "Failed to create GLib main loop" << std::endl;
            return false;
        }
        
        // Setup pipeline
        if (!setup_pipeline()) {
            std::cerr << "Failed to setup pipeline" << std::endl;
            return false;
        }
        
        // Setup recognition system
        if (!setup_recognition()) {
            std::cerr << "Failed to setup recognition system" << std::endl;
            return false;
        }
        
        // Setup MQTT (optional)
        setup_mqtt();  // Don't fail if MQTT setup fails
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

bool Application::run(int duration_ms) {
    if (!pipeline_) {
        std::cerr << "Pipeline not initialized" << std::endl;
        return false;
    }
    
    running_ = true;
    
    // Start the pipeline
    if (!pipeline_->start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return false;
    }
    
    std::cout << "Pipeline started successfully" << std::endl;
    
    // Setup duration timer if specified
    if (duration_ms > 0) {
        g_timeout_add(duration_ms, [](gpointer data) -> gboolean {
            Application* app = static_cast<Application*>(data);
            std::cout << "Duration reached, shutting down..." << std::endl;
            app->shutdown();
            return FALSE;  // Remove timer
        }, this);
    }
    
    // Run main loop
    if (loop_ && running_) {
        std::cout << "Starting main loop..." << std::endl;
        g_main_loop_run(loop_);
    }
    
    // Stop pipeline
    if (pipeline_) {
        pipeline_->stop();
    }
    
    return true;
}

void Application::shutdown() {
    if (!running_) {
        return;
    }
    
    std::cout << "Shutting down application..." << std::endl;
    running_ = false;
    
    if (loop_) {
        g_main_loop_quit(loop_);
    }
}

bool Application::setup_pipeline() {
    try {
        pipeline_ = std::make_unique<EdgeDeepStream::Pipeline>();
        
        // Set bus callback for error handling
        pipeline_->set_bus_callback([this](GstBus* bus, GstMessage* msg) -> gboolean {
            // Handle bus messages (errors, warnings, EOS, etc.)
            switch (GST_MESSAGE_TYPE(msg)) {
                case GST_MESSAGE_ERROR: {
                    GError* err = nullptr;
                    gchar* debug_info = nullptr;
                    gst_message_parse_error(msg, &err, &debug_info);
                    std::cerr << "Error from " << GST_OBJECT_NAME(msg->src) 
                             << ": " << err->message << std::endl;
                    if (debug_info) {
                        std::cerr << "Debug info: " << debug_info << std::endl;
                        g_free(debug_info);
                    }
                    g_error_free(err);
                    shutdown();
                    return FALSE;
                }
                case GST_MESSAGE_WARNING: {
                    GError* err = nullptr;
                    gchar* debug_info = nullptr;
                    gst_message_parse_warning(msg, &err, &debug_info);
                    std::cout << "Warning from " << GST_OBJECT_NAME(msg->src) 
                             << ": " << err->message << std::endl;
                    if (debug_info) {
                        std::cout << "Debug info: " << debug_info << std::endl;
                        g_free(debug_info);
                    }
                    g_error_free(err);
                    break;
                }
                case GST_MESSAGE_EOS:
                    std::cout << "End-Of-Stream reached" << std::endl;
                    shutdown();
                    return FALSE;
                case GST_MESSAGE_STATE_CHANGED: {
                    if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline_->get_pipeline())) {
                        GstState old_state, new_state, pending_state;
                        gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                        std::cout << "Pipeline state changed from " 
                                 << gst_element_state_get_name(old_state) << " to "
                                 << gst_element_state_get_name(new_state) << std::endl;
                    }
                    break;
                }
                default:
                    break;
            }
            return TRUE;  // Continue receiving messages
        });
        
        if (!pipeline_->create(config_)) {
            std::cerr << "Failed to create pipeline" << std::endl;
            return false;
        }
        
        // Add sources from config
        if (config_.has_section("source")) {
            auto& source_section = config_.sections.at("source");
            for (const auto& [id, uri] : source_section) {
                SourceInfo source_info;
                source_info.id = id;
                source_info.uri = uri;
                source_info.is_rtsp = uri.find("rtsp://") == 0;
                
                std::cout << "Adding source: " << id << " -> " << uri << std::endl;
                
                if (!pipeline_->add_source(source_info)) {
                    std::cerr << "Failed to add source: " << id << std::endl;
                    // Continue with other sources
                }
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in setup_pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool Application::setup_recognition() {
    try {
        // Load known faces
        std::string known_faces_dir = config_.get<std::string>("pipeline", "known_face_dir", "data/known_faces");
        auto known_faces = ConfigParser::load_faces(known_faces_dir);
        std::cout << "Loaded " << known_faces.size() << " known faces" << std::endl;
        
        // Setup FAISS index if available
        if (config_.has_section("recognition")) {
            // TODO: Initialize FAISS index
            // This would be implemented when FAISS integration is added
        }
        
        // Setup TensorRT inference
        // TODO: Initialize TensorRT inference engine
        // This would be implemented when TensorRT integration is added
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in setup_recognition: " << e.what() << std::endl;
        return false;
    }
}

bool Application::setup_mqtt() {
    try {
        // TODO: Initialize MQTT client
        // This would be implemented when MQTT integration is added
        std::cout << "MQTT setup (placeholder)" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in setup_mqtt: " << e.what() << std::endl;
        return false;
    }
}

// Utility functions
std::vector<std::string> load_known_faces(const std::string& path) {
    std::vector<std::string> faces;
    // TODO: Implement face loading logic
    return faces;
}

bool setup_gstreamer_plugins() {
    // Check for required GStreamer plugins
    std::vector<std::string> required_plugins = {
        "coreelements",     // queue, tee, etc.
        "playback",         // uridecodebin
        "videoparsersbad",  // h264parse, h265parse
        "nvcodec",          // nvh264dec, nvh265dec (if available)
        "rtp",              // rtph264depay, rtph265depay
        "rtsp",             // rtspsrc
        "videotestsrc",     // videotestsrc (for testing)
    };
    
    for (const auto& plugin_name : required_plugins) {
        GstPlugin* plugin = gst_plugin_load_by_name(plugin_name.c_str());
        if (!plugin) {
            std::cout << "Warning: Plugin '" << plugin_name << "' not available" << std::endl;
            // Don't fail - some plugins are optional
        } else {
            gst_object_unref(plugin);
        }
    }
    
    // Check for DeepStream plugins
    std::vector<std::string> deepstream_plugins = {
        "nvdsgst_meta",
        "nvdsgst_helper"
    };
    
    for (const auto& plugin_name : deepstream_plugins) {
        GstPlugin* plugin = gst_plugin_load_by_name(plugin_name.c_str());
        if (!plugin) {
            std::cerr << "Error: DeepStream plugin '" << plugin_name << "' not available" << std::endl;
            return false;
        }
        gst_object_unref(plugin);
    }
    
    std::cout << "GStreamer plugins setup completed" << std::endl;
    return true;
}

} // namespace EdgeDeepStream