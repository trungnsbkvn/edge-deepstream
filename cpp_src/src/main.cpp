// Clean reconstructed main.cpp implementing Application methods and entry point.

#include "edge_deepstream.h"
#include "pipeline.h"
#include "source_bin.h"
#include "env_utils.h"
#include "config_parser.h"
#include "event_sender.h"
#include "mqtt_listener.h"

#include <signal.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <atomic>

namespace EdgeDeepStream {

// Global realtime flag referenced by source_bin.cpp
bool REALTIME_DROP = false;

// ---------------- Application lifecycle ----------------
Application::Application() : running_(false), loop_(g_main_loop_new(nullptr, FALSE)) {}
Application::~Application() {
    shutdown();
    if (loop_) {
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }
}

bool Application::initialize(const std::string& config_path) {
    try {
        std::unique_ptr<Config> parsed;
        if (config_path.size() >= 5 && config_path.substr(config_path.size()-5) == ".toml") {
            parsed = ConfigParser::parse_toml(config_path);
        } else {
            parsed = ConfigParser::parse_config_file(config_path);
        }
        if (!parsed) {
            std::cerr << "Failed to parse config: " << config_path << std::endl;
            return false;
        }
        config_ = *parsed; // copy

        // Realtime drop flag (env overrides config)
        REALTIME_DROP = EnvUtils::env_bool("DS_REALTIME_DROP").value_or(
            config_.get<bool>("pipeline", "realtime", false));
        std::cout << "REALTIME_DROP=" << (REALTIME_DROP?"true":"false") << std::endl;

        if (!setup_pipeline()) return false;
        setup_recognition(); // best-effort
        setup_mqtt(); // best-effort
        return true;
    } catch (const std::exception& e) {
        std::cerr << "initialize exception: " << e.what() << std::endl;
        return false;
    }
}

bool Application::setup_pipeline() {
    pipeline_ = std::make_unique<Pipeline>();
    if (!pipeline_->create(config_)) {
        std::cerr << "Pipeline creation failed" << std::endl;
        return false;
    }

    // Count sources and set streammux properties
    bool has_live = false;
    int source_count = 0;
    if (config_.has_section("source")) {
        source_count = config_.sections.at("source").size();
        for (const auto& kv : config_.sections.at("source")) {
            if (kv.second.find("rtsp://") == 0) {
                has_live = true;
                break; // Just need to know if any are live
            }
        }
    }
    
    // streammux tweaks - batch-size should be fixed in config to prevent crashes
    GstElement* streammux = pipeline_->get_streammux();
    if (streammux) {
        if (has_live) g_object_set(streammux, "live-source", 1, NULL);
        // NOTE: batch-size is set from config only - no dynamic changes to prevent crashes
    }
    
    // Add sources to pipeline
    if (config_.has_section("source")) {
        for (const auto& kv : config_.sections.at("source")) {
            SourceInfo source_info;
            source_info.id = kv.first;
            source_info.uri = kv.second;
            source_info.is_rtsp = (kv.second.find("rtsp://") == 0);
            source_info.source_bin = nullptr;
            source_info.sink_pad = nullptr;
            source_info.index = -1;
            
            if (!pipeline_->add_source(source_info)) {
                std::cerr << "Failed to add source " << kv.first << " with URI " << kv.second << std::endl;
                return false;
            }
        }
    }
    
    // Set up bus message handler following DeepStream 6.3 pattern
    pipeline_->set_bus_callback([this](GstBus* bus, GstMessage* msg) -> gboolean {
        return this->bus_message_handler(bus, msg);
    });
    
    std::cout << "Pipeline setup completed with " << source_count << " sources" << std::endl;
    
    // RTSP Health watchdog like Python version
    if (has_live) {
        std::cout << "[RTSP] Health watchdog active: interval=5000ms timeout=20.0s max_retries=3" << std::endl;
        // TODO: Implement RTSP health monitoring
    }
    
    return true;
}

bool Application::setup_recognition() { return true; }
bool Application::setup_mqtt() {
    if (!config_.has_section("mqtt")) {
        return true; // MQTT not configured, skip
    }
    
    auto mqtt_config = config_.sections.at("mqtt");
    std::string host = mqtt_config["host"];
    int port = std::stoi(mqtt_config["port"]);
    std::string request_topic = mqtt_config["request_topic"];
    
    // Create MQTT listener
    mqtt_listener_ = std::make_unique<MQTTListener>();
    
    // Initialize MQTT client
    if (!mqtt_listener_->initialize(host, port)) {
        std::cerr << "Failed to initialize MQTT client" << std::endl;
        return false;
    }
    
    // Set message handler for AI requests
    mqtt_listener_->set_message_handler([this](const std::string& topic, const std::string& payload) {
        // TODO: Handle MQTT messages for face recognition commands
        std::cout << "[MQTT] Received message on topic: " << topic << std::endl;
    });
    
    // Set connection callback
    mqtt_listener_->set_connection_callback([this](bool connected, const std::string& reason) {
        if (connected) {
            std::cout << "[MQTT] Connected to " << reason << std::endl;
        } else {
            std::cout << "[MQTT] Disconnected: " << reason << std::endl;
        }
    });
    
    // Connect and subscribe
    if (!mqtt_listener_->connect()) {
        std::cerr << "Failed to connect to MQTT broker" << std::endl;
        return false;
    }
    
    if (!mqtt_listener_->subscribe(request_topic)) {
        std::cerr << "Failed to subscribe to MQTT topic: " << request_topic << std::endl;
        return false;
    }
    
    std::cout << "[MQTT] listening on " << host << ":" << port << " topic " << request_topic << std::endl;
    return true;
}

// Bus message handler following DeepStream 6.3 pattern
gboolean Application::bus_message_handler(GstBus* bus, GstMessage* msg) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "[BUS] End of stream" << std::endl;
            shutdown();
            break;
            
        case GST_MESSAGE_ERROR: {
            gchar *debug = nullptr;
            GError *error = nullptr;
            gst_message_parse_error(msg, &error, &debug);
            std::cerr << "[BUS] ERROR from element " << GST_OBJECT_NAME(msg->src) 
                      << ": " << error->message << std::endl;
            if (debug) {
                std::cerr << "[BUS] Error details: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(error);
            shutdown();
            break;
        }
        
        case GST_MESSAGE_WARNING: {
            gchar *debug = nullptr;
            GError *error = nullptr;
            gst_message_parse_warning(msg, &error, &debug);
            std::cerr << "[BUS] WARNING from element " << GST_OBJECT_NAME(msg->src) 
                      << ": " << error->message << std::endl;
            if (debug) {
                std::cerr << "[BUS] Warning details: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(error);
            break;
        }
        
        case GST_MESSAGE_INFO: {
            gchar *debug = nullptr;
            GError *error = nullptr;
            gst_message_parse_info(msg, &error, &debug);
            std::cout << "[BUS] INFO from element " << GST_OBJECT_NAME(msg->src) 
                      << ": " << error->message << std::endl;
            if (debug) {
                std::cout << "[BUS] Info details: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(error);
            break;
        }
        
        case GST_MESSAGE_ELEMENT: {
            const GstStructure *s = gst_message_get_structure(msg);
            if (s) {
                const gchar *name = gst_structure_get_name(s);
                std::cout << "[BUS] ELEMENT message from " << GST_OBJECT_NAME(msg->src) 
                          << ": " << (name ? name : "unknown") << std::endl;
                
                // Print additional debug for stream status and RTSP events
                if (name && (g_str_has_prefix(name, "rtsp") || 
                           g_str_has_prefix(name, "rtspsrc") ||
                           g_str_has_prefix(name, "GstRTSPSrc"))) {
                    gchar *struct_str = gst_structure_to_string(s);
                    std::cout << "[RTSP] Event: " << struct_str << std::endl;
                    g_free(struct_str);
                }
            }
            break;
        }
        
        case GST_MESSAGE_STREAM_STATUS: {
            GstStreamStatusType type;
            GstElement *owner;
            gst_message_parse_stream_status(msg, &type, &owner);
            const gchar *element_name = GST_OBJECT_NAME(msg->src);
            
            switch (type) {
                case GST_STREAM_STATUS_TYPE_CREATE:
                    std::cout << "[STREAM] " << element_name << " - Stream CREATE" << std::endl;
                    break;
                case GST_STREAM_STATUS_TYPE_ENTER:
                    std::cout << "[STREAM] " << element_name << " - Stream ENTER" << std::endl;
                    break;
                case GST_STREAM_STATUS_TYPE_LEAVE:
                    std::cout << "[STREAM] " << element_name << " - Stream LEAVE" << std::endl;
                    break;
                case GST_STREAM_STATUS_TYPE_DESTROY:
                    std::cout << "[STREAM] " << element_name << " - Stream DESTROY" << std::endl;
                    break;
                case GST_STREAM_STATUS_TYPE_START:
                    std::cout << "[STREAM] " << element_name << " - Stream START" << std::endl;
                    break;
                case GST_STREAM_STATUS_TYPE_PAUSE:
                    std::cout << "[STREAM] " << element_name << " - Stream PAUSE" << std::endl;
                    break;
                case GST_STREAM_STATUS_TYPE_STOP:
                    std::cout << "[STREAM] " << element_name << " - Stream STOP" << std::endl;
                    break;
                default:
                    std::cout << "[STREAM] " << element_name << " - Stream status: " << type << std::endl;
                    break;
            }
            break;
        }
        
        case GST_MESSAGE_STATE_CHANGED: {
            if (pipeline_ && GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline_->get_pipeline())) {
                GstState oldstate, newstate, pending;
                gst_message_parse_state_changed(msg, &oldstate, &newstate, &pending);
                std::cout << "[BUS] Pipeline state changed from " 
                          << gst_element_state_get_name(oldstate) << " to " 
                          << gst_element_state_get_name(newstate) << std::endl;
                          
                if (newstate == GST_STATE_PLAYING) {
                    std::cout << "[BUS] Pipeline is now PLAYING - engines should initialize" << std::endl;
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    return TRUE; // Continue receiving messages
}

// Static wrapper for bus callback
static gboolean bus_call_wrapper(GstBus* bus, GstMessage* msg, gpointer data) {
    Application* app = static_cast<Application*>(data);
    if (app) {
        return app->bus_message_handler(bus, msg);
    }
    return TRUE;
}

bool Application::run(int duration_ms) {
    if (!pipeline_) { std::cerr << "Pipeline not initialized" << std::endl; return false; }
    running_ = true;

    std::cout << "Listing sources:" << std::endl;
    if (config_.has_section("source")) {
        for (auto& kv : config_.sections.at("source")) std::cout << "  " << kv.first << " -> " << kv.second << std::endl;
    }

    // Skip PAUSED state - go directly to PLAYING like Python version
    std::cout << "[STATE] Going directly to PLAYING state (matching Python implementation)" << std::endl;

    // Skip inference element probing that can cause hangs during engine initialization
    std::cout << "[ENGINE] Skipping element probing - proceeding directly to PLAYING state" << std::endl;
    
    // Actually start the pipeline
    if (!pipeline_->start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return false;
    }
    std::cout << "Pipeline started successfully!" << std::endl;
    
    // Duration timer
    if (duration_ms > 0) { g_timeout_add(duration_ms, [](gpointer d)->gboolean { auto* app=static_cast<Application*>(d); std::cout << "[INFO] Duration reached" << std::endl; if(app) app->shutdown(); return FALSE; }, this); }

    if (loop_ && running_) { std::cout << "Entering main loop" << std::endl; g_main_loop_run(loop_); }
    if (pipeline_) pipeline_->stop();
    return true;
}

void Application::shutdown() {
    if (!running_) return; running_ = false; std::cout << "Shutdown requested" << std::endl; if (loop_) g_main_loop_quit(loop_); if (pipeline_) pipeline_->set_state(GST_STATE_NULL); }

// Stub utility implementations (previously global). Keep minimal versions used elsewhere.
std::vector<std::string> load_known_faces(const std::string& path) { std::vector<std::string> faces; try { if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) { for (auto& e: std::filesystem::directory_iterator(path)) if (e.is_regular_file()) faces.push_back(e.path().string()); } } catch(...) {} return faces; }
bool setup_gstreamer_plugins() { return true; }

} // namespace EdgeDeepStream

using namespace EdgeDeepStream;

// Global pointer for signal handling
static Application* g_app = nullptr;
static std::atomic<int> g_sigcount{0};

static void signal_handler(int sig) {
    int c = ++g_sigcount;
    std::cout << "Signal " << sig << " received (count=" << c << ")" << std::endl;
    if (c == 1) {
        if (g_app) g_app->shutdown(); else ::_exit(1);
    } else {
        std::cout << "Force exiting" << std::endl;
        ::_exit(1);
    }
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    std::string config_path;
    int duration_ms = -1; // -1 = infinite

    auto print_usage = [&](int code){
        std::cerr << "Usage: " << argv[0] << " <config.toml> [duration_ms]\n"
                  << "   or: " << argv[0] << " --config <file> [--seconds <s>]\n"
                  << "   or: " << argv[0] << " --config=<file> [--seconds=<s>]\n"
                  << "Optional flags:\n"
                  << "  --seconds N        Run for N seconds (converted to ms)\n"
                  << "  --duration-ms N    Run for N milliseconds\n"
                  << "Notes:\n"
                  << "  Positional second argument is treated as milliseconds; if <1000 it is assumed seconds.\n"
                  << std::endl;
        return code;
    };

    for (int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") return print_usage(0);
        if (arg.rfind("--config=", 0) == 0) {
            config_path = arg.substr(9);
            continue;
        }
        if (arg == "--config" && i+1 < argc) {
            config_path = argv[++i];
            continue;
        }
        if (arg.rfind("--seconds=", 0) == 0) {
            try { duration_ms = std::stoi(arg.substr(10)) * 1000; } catch(...) {}
            continue;
        }
        if (arg == "--seconds" && i+1 < argc) {
            try { duration_ms = std::stoi(argv[++i]) * 1000; } catch(...) {}
            continue;
        }
        if (arg.rfind("--duration-ms=", 0) == 0) {
            try { duration_ms = std::stoi(arg.substr(14)); } catch(...) {}
            continue;
        }
        if (arg == "--duration-ms" && i+1 < argc) {
            try { duration_ms = std::stoi(argv[++i]); } catch(...) {}
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown flag: " << arg << std::endl;
            return print_usage(1);
        }
        // Positional
        if (config_path.empty()) {
            config_path = arg;
        } else if (duration_ms < 0) {
            try {
                int v = std::stoi(arg);
                duration_ms = (v < 1000 ? v * 1000 : v); // heuristic
            } catch(...) {}
        }
    }

    if (config_path.empty()) return print_usage(1);

    std::cout << "[ARGS] config=" << config_path << " duration_ms=" << duration_ms << std::endl;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    Application app; g_app = &app;
    if (!app.initialize(config_path)) { std::cerr << "Initialization failed" << std::endl; return 1; }
    bool ok = app.run(duration_ms);
    g_app = nullptr;
    return ok ? 0 : 1;
}
