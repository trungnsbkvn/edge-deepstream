#include "pipeline.h"
#include "source_bin.h"
#include "config_parser.h"
#include "env_utils.h"
#include "probe.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <unistd.h>

namespace EdgeDeepStream {

Pipeline::Pipeline() 
    : pipeline_(nullptr), streammux_(nullptr), pgie_(nullptr), sgie_(nullptr),
      tracker_(nullptr), tiler_(nullptr), nvvidconv_(nullptr), osd_(nullptr), sink_(nullptr),
      queue1_(nullptr), queue2_(nullptr), queue3_(nullptr), queue4_(nullptr), queue5_(nullptr),
      queue6_(nullptr), queue7_(nullptr),
      bus_(nullptr), bus_watch_id_(0), next_source_index_(0), frame_count_(0) {
}

Pipeline::~Pipeline() {
    destroy();
}

bool Pipeline::create(const Config& config) {
    try {
        // Store config for later use
        config_ = config;
        
        // Create main pipeline
        pipeline_ = gst_pipeline_new("deepstream-pipeline");
        if (!pipeline_) {
            std::cerr << "Failed to create pipeline" << std::endl;
            return false;
        }
        
        // Create all elements
        if (!create_elements(config)) {
            std::cerr << "Failed to create pipeline elements" << std::endl;
            return false;
        }
        
    // Add elements to pipeline - only add elements that were created
    // ADDING PGIE/TRACKER: add streammux, queue1, pgie, queue2, tracker, queue3, tiler, nvvidconv, osd, sink
    if (!gst_bin_add(GST_BIN(pipeline_), streammux_)) {
        std::cerr << "Failed to add streammux to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), queue1_)) {
        std::cerr << "Failed to add queue1 to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), pgie_)) {
        std::cerr << "Failed to add pgie to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), queue2_)) {
        std::cerr << "Failed to add queue2 to pipeline" << std::endl;
        return false;
    }
    // if (!gst_bin_add(GST_BIN(pipeline_), tracker_)) {
    //     std::cerr << "Failed to add tracker to pipeline" << std::endl;
    //     return false;
    // }
    // if (!gst_bin_add(GST_BIN(pipeline_), queue3_)) {
    //     std::cerr << "Failed to add queue3 to pipeline" << std::endl;
    //     return false;
    // }
    // sgie_, queue4_ are commented out
    if (!gst_bin_add(GST_BIN(pipeline_), tiler_)) {
        std::cerr << "Failed to add tiler to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), queue5_)) {
        std::cerr << "Failed to add queue5 to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), nvvidconv_)) {
        std::cerr << "Failed to add nvvidconv to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), queue6_)) {
        std::cerr << "Failed to add queue6 to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), osd_)) {
        std::cerr << "Failed to add osd to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), queue7_)) {
        std::cerr << "Failed to add queue7 to pipeline" << std::endl;
        return false;
    }
    if (!gst_bin_add(GST_BIN(pipeline_), sink_)) {
        std::cerr << "Failed to add sink to pipeline" << std::endl;
        return false;
    }        // Link elements
        if (!link_elements()) {
            std::cerr << "Failed to link pipeline elements" << std::endl;
            return false;
        }
        
        // Set element properties
        if (!setup_element_properties(config)) {
            std::cerr << "Failed to setup element properties" << std::endl;
            return false;
        }
        
    // Setup bus callback
    bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    bus_watch_id_ = gst_bus_add_watch(bus_, bus_call_wrapper, this);
    gst_object_unref(bus_);
    
    // Create alignment directory like Python version
    std::filesystem::create_directories("/dev/shm/edge-deepstream/aligned");
    std::cout << "[PIPELINE] Alignment directory ensured: /dev/shm/edge-deepstream/aligned" << std::endl;
    
    std::cout << "Pipeline created successfully" << std::endl;
    return true;    } catch (const std::exception& e) {
        std::cerr << "Exception in create_pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool Pipeline::start() {
    if (!pipeline_) {
        std::cerr << "Pipeline not created" << std::endl;
        return false;
    }
    
    std::cout << "Starting pipeline..." << std::endl;
    start_time_ = std::chrono::steady_clock::now();
    
    // Follow Python version initial start sequence: set inference to READY, then pipeline directly to PLAYING
    std::cout << "[STATE] Following Python initial start: inference READY -> pipeline PLAYING" << std::endl;
    
    // Step 1: Set inference elements to READY state (like Python version)
    if (pgie_) {
        std::cout << "[STATE] Setting PGIE to READY..." << std::endl;
        gst_element_set_state(pgie_, GST_STATE_READY);
    }
    
    // if (tracker_) {
    //     std::cout << "[STATE] Setting tracker to READY..." << std::endl;
    //     gst_element_set_state(tracker_, GST_STATE_READY);
    // }
    
    // Small delay like Python version
    g_usleep(200000); // 200ms delay
    
    // Step 2: Set inference elements to PLAYING state
    if (pgie_) {
        std::cout << "[STATE] Setting PGIE to PLAYING..." << std::endl;
        gst_element_set_state(pgie_, GST_STATE_PLAYING);
    }
    
    if (tracker_) {
        std::cout << "[STATE] Setting tracker to PLAYING..." << std::endl;
        gst_element_set_state(tracker_, GST_STATE_PLAYING);
    }
    
    // Step 3: Set pipeline directly to PLAYING state (like Python initial start)
    std::cout << "[STATE] Setting pipeline directly to PLAYING state..." << std::endl;
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    std::cout << "Pipeline state change result: " << ret << std::endl;
    
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Unable to set pipeline to playing state" << std::endl;
        return false;
    }
    
    // If state change is asynchronous, wait for it to complete
    if (ret == GST_STATE_CHANGE_ASYNC) {
        std::cout << "Waiting for asynchronous state change to complete..." << std::endl;
        GstState current_state, pending_state;
        // Wait up to 15 seconds for the state change to complete (longer for RTSP + inference)
        ret = gst_element_get_state(pipeline_, &current_state, &pending_state, 15 * GST_SECOND);
        std::cout << "State change completion result: " << ret << std::endl;
        
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to complete state change to playing" << std::endl;
            return false;
        }
    }
    
    // Check actual state after transition
    GstState current_state, pending_state;
    gst_element_get_state(pipeline_, &current_state, &pending_state, GST_CLOCK_TIME_NONE);
    std::cout << "Pipeline actual state: " << gst_element_state_get_name(current_state) 
              << " (pending: " << gst_element_state_get_name(pending_state) << ")" << std::endl;
    
    if (current_state != GST_STATE_PLAYING) {
        std::cerr << "Pipeline failed to reach PLAYING state, current state: " 
                  << gst_element_state_get_name(current_state) << std::endl;
        return false;
    }
    
    // Attach probes to inference elements
    if (!attach_probes(pgie_, sgie_, nullptr)) {
        std::cerr << "Failed to attach probes to pipeline elements" << std::endl;
        return false;
    }
    
    return true;
}

bool Pipeline::stop() {
    if (!pipeline_) {
        return true;
    }
    
    std::cout << "Stopping pipeline..." << std::endl;
    
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Unable to set pipeline to null state" << std::endl;
        return false;
    }
    
    return true;
}

void Pipeline::destroy() {
    if (bus_watch_id_) {
        g_source_remove(bus_watch_id_);
        bus_watch_id_ = 0;
    }
    
    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    
    // Clear element pointers (they're freed when pipeline is destroyed)
    streammux_ = nullptr;
    pgie_ = nullptr;
    sgie_ = nullptr;
    tracker_ = nullptr;
    tiler_ = nullptr;
    nvvidconv_ = nullptr;
    osd_ = nullptr;
    sink_ = nullptr;
    queue1_ = queue2_ = queue3_ = queue4_ = queue5_ = queue6_ = queue7_ = nullptr;
    
    sources_.clear();
}

bool Pipeline::set_state(GstState state) {
    if (!pipeline_) {
        return false;
    }
    
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, state);
    return ret != GST_STATE_CHANGE_FAILURE;
}

GstState Pipeline::get_state() {
    if (!pipeline_) {
        return GST_STATE_NULL;
    }
    
    GstState state, pending;
    GstStateChangeReturn ret = gst_element_get_state(pipeline_, &state, &pending, GST_CLOCK_TIME_NONE);
    
    if (ret == GST_STATE_CHANGE_SUCCESS) {
        return state;
    }
    
    return GST_STATE_NULL;
}

bool Pipeline::add_source(const SourceInfo& source) {
    try {
        // Create source bin
        auto source_bin = std::make_unique<SourceBin>(next_source_index_, source.uri);
        if (!source_bin->create()) {
            std::cerr << "Failed to create source bin for " << source.uri << std::endl;
            return false;
        }
        
        // Add source bin to pipeline
        if (!gst_bin_add(GST_BIN(pipeline_), source_bin->get_bin())) {
            std::cerr << "Failed to add source bin to pipeline" << std::endl;
            return false;
        }
        
        // Get sink pad from streammux
        gchar pad_name[16];
        g_snprintf(pad_name, 16, "sink_%u", next_source_index_);
        GstPad* sinkpad = gst_element_get_request_pad(streammux_, pad_name);
        if (!sinkpad) {
            std::cerr << "Failed to get sink pad from streammux" << std::endl;
            return false;
        }
        
        // Get src pad from source bin
        GstPad* srcpad = gst_element_get_static_pad(source_bin->get_bin(), "src");
        if (!srcpad) {
            std::cerr << "Failed to get src pad from source bin" << std::endl;
            gst_object_unref(sinkpad);
            return false;
        }
        
        // Link source bin to streammux
        GstPadLinkReturn ret = gst_pad_link(srcpad, sinkpad);
        if (ret != GST_PAD_LINK_OK) {
            std::cerr << "Failed to link source bin to streammux: " << ret << std::endl;
            gst_object_unref(srcpad);
            gst_object_unref(sinkpad);
            return false;
        }
        
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
        
        // Sync state with parent
        source_bin->set_state(get_state());
        
        // Store source info
        SourceInfo stored_source = source;
        stored_source.index = next_source_index_;
        stored_source.source_bin = source_bin->get_bin();
        sources_[source.id] = stored_source;
        
        // Release ownership of source_bin (it's now managed by the pipeline)
        source_bin.release();
        
        next_source_index_++;
        
        std::cout << "Added source " << source.id << " at index " << stored_source.index << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in add_source: " << e.what() << std::endl;
        return false;
    }
}

bool Pipeline::remove_source(const std::string& source_id) {
    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        std::cerr << "Source not found: " << source_id << std::endl;
        return false;
    }
    
    const SourceInfo& source = it->second;
    
    // TODO: Implement source removal
    // This requires careful handling of pad unlinking and element removal
    
    sources_.erase(it);
    std::cout << "Removed source: " << source_id << std::endl;
    return true;
}

void Pipeline::set_bus_callback(BusMessageCallback callback) {
    bus_callback_ = callback;
}

void Pipeline::update_performance_stats() {
    frame_count_++;
    
    if (frame_count_ % 1000 == 0) {  // Log every 1000 frames
        auto current_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_);
        
        if (duration.count() > 0) {
            double fps = (frame_count_ * 1000.0) / duration.count();
            std::cout << "Pipeline FPS: " << fps << " (frames: " << frame_count_ << ")" << std::endl;
        }
    }
}

bool Pipeline::create_elements(const Config& config) {
    // Create streammux
    streammux_ = create_streammux(config);
    if (!streammux_) {
        std::cerr << "Failed to create streammux" << std::endl;
        return false;
    }
    
    // Create queues
    queue1_ = gst_element_factory_make("queue", "queue1");
    queue2_ = gst_element_factory_make("queue", "queue2");
    queue3_ = gst_element_factory_make("queue", "queue3");
    queue4_ = gst_element_factory_make("queue", "queue4");
    queue5_ = gst_element_factory_make("queue", "queue5");
    queue6_ = gst_element_factory_make("queue", "queue6");
    queue7_ = gst_element_factory_make("queue", "queue7");
    
    if (!queue1_ || !queue2_ || !queue3_ || !queue4_ || !queue5_ || !queue6_ || !queue7_) {
        std::cerr << "Failed to create queue elements" << std::endl;
        return false;
    }
    
    // Create PGIE (Primary GStreamer Inference Engine)
    pgie_ = create_pgie(config);
    if (!pgie_) {
        std::cerr << "Failed to create pgie" << std::endl;
        return false;
    }
    
    // Create tracker
    // tracker_ = create_tracker(config);
    // if (!tracker_) {
    //     std::cerr << "Failed to create tracker" << std::endl;
    //     return false;
    // }
    
    // Create SGIE (Secondary GStreamer Inference Engine) - COMMENTED OUT FOR TESTING
    // sgie_ = create_sgie(config);
    // if (!sgie_) {
    //     std::cerr << "Failed to create sgie" << std::endl;
    //     return false;
    // }
    
    // Create tiler
    tiler_ = create_tiler(config);
    if (!tiler_) {
        std::cerr << "Failed to create tiler" << std::endl;
        return false;
    }
    
    // Create nvvidconv (video converter)
    nvvidconv_ = create_nvvidconv(config);
    if (!nvvidconv_) {
        std::cerr << "Failed to create nvvidconv" << std::endl;
        return false;
    }
    
    // Create OSD (On-Screen Display)
    osd_ = create_osd(config);
    if (!osd_) {
        std::cerr << "Failed to create osd" << std::endl;
        return false;
    }
    
    // Create sink
    sink_ = create_sink(config);
    if (!sink_) {
        std::cerr << "Failed to create sink" << std::endl;
        return false;
    }
    
    return true;
}

bool Pipeline::link_elements() {
    // Check display mode
    int display = config_.get<int>("pipeline", "display", 0);
    
    // Full pipeline flow: streammux -> queue1 -> pgie -> queue2 -> tracker -> queue3 -> sgie -> queue4 -> tiler -> queue5 -> nvvidconv -> queue6
    // ADDING PGIE/TRACKER: streammux -> queue1 -> pgie -> queue2 -> tracker -> queue3 -> tiler -> queue5 -> nvvidconv -> queue6
    
    if (!gst_element_link_many(streammux_, queue1_, pgie_, queue2_, tiler_, queue5_, nvvidconv_, queue6_, NULL)) {
        std::cerr << "Failed to link main pipeline elements" << std::endl;
        return false;
    }
    
    if (display) {
        // Display mode: queue6 -> osd -> queue7 -> sink
        if (!gst_element_link_many(queue6_, osd_, queue7_, sink_, NULL)) {
            std::cerr << "Failed to link display pipeline elements" << std::endl;
            return false;
        }
    } else {
        // Headless mode: bypass OSD for better throughput - queue6 -> sink
        if (!gst_element_link(queue6_, sink_)) {
            std::cerr << "Failed to link headless pipeline elements" << std::endl;
            return false;
        }
    }
    
    std::cout << "Pipeline elements linked successfully" << std::endl;
    return true;
}

bool Pipeline::setup_element_properties(const Config& config) {
    // Setup streammux properties
    if (config.has_section("streammux")) {
        ConfigParser::set_element_properties(streammux_, config.sections.at("streammux"));
    }
    
    // PGIE properties
    if (config.has_section("pgie")) {
        auto pgie_config = config.sections.at("pgie");
        auto config_file_it = pgie_config.find("config-file-path");
        if (config_file_it != pgie_config.end()) {
            ConfigParser::set_element_properties(pgie_, pgie_config);
        }
    }
    
    // SGIE properties
    if (config.has_section("sgie")) {
        auto sgie_config = config.sections.at("sgie");
        auto config_file_it = sgie_config.find("config-file-path");
        if (config_file_it != sgie_config.end()) {
            ConfigParser::set_element_properties(sgie_, sgie_config);
        }
    }
    
    // Tracker properties
    if (config.has_section("tracker")) {
        auto tracker_config = config.sections.at("tracker");
        auto config_file_it = tracker_config.find("config-file-path");
        if (config_file_it != tracker_config.end()) {
            ConfigParser::set_tracker_properties(tracker_, config_file_it->second);
        }
    }
    
    // Setup tiler properties
    if (config.has_section("tiler")) {
        ConfigParser::set_element_properties(tiler_, config.sections.at("tiler"));
    }
    
    // Setup OSD properties
    if (config.has_section("nvosd")) {
        ConfigParser::set_element_properties(osd_, config.sections.at("nvosd"));
    }
    
    // Setup sink properties
    if (config.has_section("sink")) {
        ConfigParser::set_element_properties(sink_, config.sections.at("sink"));
    }
    
    return true;
}

GstElement* Pipeline::create_streammux(const Config& config) {
    GstElement* element = gst_element_factory_make("nvstreammux", "streammux");
    if (!element) {
        std::cerr << "Failed to create nvstreammux element" << std::endl;
        return nullptr;
    }
    
    // Set default properties
    g_object_set(G_OBJECT(element),
                 "batch-size", 8,
                 "batched-push-timeout", 15000,
                 "width", 960,
                 "height", 540,
                 "live-source", 1,
                 NULL);
    
    return element;
}

GstElement* Pipeline::create_pgie(const Config& config) {
    GstElement* element = gst_element_factory_make("nvinfer", "pgie");
    if (!element) {
        std::cerr << "Failed to create nvinfer element for pgie" << std::endl;
        return nullptr;
    }
    
    // Properties will be set later by setup_element_properties
    // No need to set them here to avoid double-setting
    
    return element;
}

GstElement* Pipeline::create_sgie(const Config& config) {
    GstElement* element = gst_element_factory_make("nvinfer", "sgie");
    if (!element) {
        std::cerr << "Failed to create nvinfer element for sgie" << std::endl;
        return nullptr;
    }
    
    // Properties will be set later by setup_element_properties
    // No need to set them here to avoid double-setting
    
    return element;
}

GstElement* Pipeline::create_tracker(const Config& config) {
    GstElement* element = gst_element_factory_make("nvtracker", "tracker");
    if (!element) {
        std::cerr << "Failed to create nvtracker element" << std::endl;
        return nullptr;
    }
    
    return element;
}

GstElement* Pipeline::create_tiler(const Config& config) {
    GstElement* element = gst_element_factory_make("nvmultistreamtiler", "tiler");
    if (!element) {
        std::cerr << "Failed to create nvmultistreamtiler element" << std::endl;
        return nullptr;
    }
    
    // Calculate grid size based on number of sources
    // We have 7 sources, so use 3x3 grid (9 tiles)
    int rows = 3;
    int columns = 3;
    
    // Get dimensions from config or use defaults
    int width = config.get<int>("tiler", "width", 1280);
    int height = config.get<int>("tiler", "height", 720);
    
    // Set properties
    g_object_set(G_OBJECT(element),
                 "rows", rows,
                 "columns", columns,
                 "width", width,
                 "height", height,
                 NULL);
    
    std::cout << "[TILER] Configured " << rows << "x" << columns << " grid (" << width << "x" << height << ")" << std::endl;
    
    return element;
}

GstElement* Pipeline::create_nvvidconv(const Config& config) {
    GstElement* element = gst_element_factory_make("nvvideoconvert", "nvvidconv");
    if (!element) {
        std::cerr << "Failed to create nvvideoconvert element" << std::endl;
        return nullptr;
    }
    
    // Configure for display output - convert NVMM to RGBA
    g_object_set(G_OBJECT(element),
                 "nvbuf-memory-type", 0,  // Use system memory for output
                 NULL);
    
    return element;
}

GstElement* Pipeline::create_osd(const Config& config) {
    GstElement* element = gst_element_factory_make("nvdsosd", "osd");
    if (!element) {
        std::cerr << "Failed to create nvdsosd element" << std::endl;
        return nullptr;
    }
    
    // Set default properties
    g_object_set(G_OBJECT(element),
                 "process-mode", 0,
                 "display-text", 1,
                 NULL);
    
    return element;
}

GstElement* Pipeline::create_sink(const Config& config) {
    // Check if display is enabled
    int display = config.get<int>("pipeline", "display", 0);
    
    // Check for GUI display like Python version - more robust detection
    const char* display_env = getenv("DISPLAY");
    const char* wayland_env = getenv("WAYLAND_DISPLAY");
    bool has_display_env = (display_env && strlen(display_env) > 0) || (wayland_env && strlen(wayland_env) > 0);
    
    // Additional check: try to detect if we're in a graphical environment
    // by checking if we can access X11 or Wayland sockets
    bool has_graphical_env = has_display_env;
    if (!has_graphical_env) {
        // Check for X11 socket
        if (access("/tmp/.X11-unix", F_OK) == 0) {
            has_graphical_env = true;
        }
        // Check for Wayland socket (common locations)
        else if (access("/run/user/1000/wayland-0", F_OK) == 0 || 
                 access("/tmp/wayland-0", F_OK) == 0) {
            has_graphical_env = true;
        }
    }
    
    GstElement* element;
    if (display && has_graphical_env) {
        // GUI display available - choose sink based on architecture like Python version
        #ifdef __aarch64__
            // On Jetson (aarch64), use nv3dsink
            std::cout << "Creating nv3dsink for Jetson device" << std::endl;
            element = gst_element_factory_make("nv3dsink", "sink");
            if (!element) {
                std::cerr << "Unable to create nv3dsink" << std::endl;
                element = gst_element_factory_make("nveglglessink", "sink");
            }
        #else
            // On x86, use nveglglessink with fallbacks
            element = gst_element_factory_make("nveglglessink", "sink");
            if (!element) {
                element = gst_element_factory_make("xvimagesink", "sink");
            }
            if (!element) {
                element = gst_element_factory_make("ximagesink", "sink");
            }
            if (!element) {
                element = gst_element_factory_make("autovideosink", "sink");
            }
        #endif
    } else {
        // No GUI display or display disabled - use fakesink
        if (display && !has_graphical_env) {
            std::cout << "No GUI display detected. Falling back to fakesink for headless run." << std::endl;
        }
        element = gst_element_factory_make("fakesink", "sink");
    }
    
    if (element && gst_element_factory_make("fakesink", "sink") == element) {
        // Configure fakesink
        g_object_set(G_OBJECT(element),
                     "sync", 0,
                     "enable-last-sample", 0,
                     NULL);
    } else if (element) {
        // Configure display sink
        g_object_set(G_OBJECT(element),
                     "sync", 0,
                     "qos", 0,
                     NULL);
    }
    
    if (!element) {
        std::cerr << "Failed to create sink element" << std::endl;
        return nullptr;
    }
    
    return element;
}

gboolean Pipeline::bus_call_wrapper(GstBus* bus, GstMessage* msg, gpointer data) {
    Pipeline* pipeline = static_cast<Pipeline*>(data);
    if (pipeline && pipeline->bus_callback_) {
        return pipeline->bus_callback_(bus, msg);
    }
    return TRUE;
}

} // namespace EdgeDeepStream