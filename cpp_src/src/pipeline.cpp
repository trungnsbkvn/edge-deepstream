#include "pipeline.h"
#include "source_bin.h"
#include "config_parser.h"
#include "env_utils.h"
#include <iostream>
#include <memory>

namespace EdgeDeepStream {

Pipeline::Pipeline() 
    : pipeline_(nullptr), streammux_(nullptr), pgie_(nullptr), sgie_(nullptr),
      tracker_(nullptr), tiler_(nullptr), osd_(nullptr), sink_(nullptr),
      queue1_(nullptr), queue2_(nullptr), queue3_(nullptr), queue4_(nullptr), queue5_(nullptr),
      bus_(nullptr), bus_watch_id_(0), next_source_index_(0), frame_count_(0) {
}

Pipeline::~Pipeline() {
    destroy();
}

bool Pipeline::create(const Config& config) {
    try {
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
        
        // Add elements to pipeline
        gst_bin_add_many(GST_BIN(pipeline_), 
                         streammux_, queue1_, pgie_, queue2_,
                         tracker_, queue3_, sgie_, queue4_,
                         tiler_, osd_, queue5_, sink_, NULL);
        
        // Link elements
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
        
        std::cout << "Pipeline created successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
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
    
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Unable to set pipeline to playing state" << std::endl;
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
    osd_ = nullptr;
    sink_ = nullptr;
    queue1_ = queue2_ = queue3_ = queue4_ = queue5_ = nullptr;
    
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
    
    if (!queue1_ || !queue2_ || !queue3_ || !queue4_ || !queue5_) {
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
    tracker_ = create_tracker(config);
    if (!tracker_) {
        std::cerr << "Failed to create tracker" << std::endl;
        return false;
    }
    
    // Create SGIE (Secondary GStreamer Inference Engine)
    sgie_ = create_sgie(config);
    if (!sgie_) {
        std::cerr << "Failed to create sgie" << std::endl;
        return false;
    }
    
    // Create tiler
    tiler_ = create_tiler(config);
    if (!tiler_) {
        std::cerr << "Failed to create tiler" << std::endl;
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
    // Link: streammux -> queue1 -> pgie -> queue2 -> tracker -> queue3 -> sgie -> queue4 -> tiler -> osd -> queue5 -> sink
    
    if (!gst_element_link_many(streammux_, queue1_, pgie_, queue2_, 
                              tracker_, queue3_, sgie_, queue4_,
                              tiler_, osd_, queue5_, sink_, NULL)) {
        std::cerr << "Failed to link pipeline elements" << std::endl;
        return false;
    }
    
    std::cout << "Pipeline elements linked successfully" << std::endl;
    return true;
}

bool Pipeline::setup_element_properties(const Config& config) {
    // Setup streammux properties
    if (config.has_section("streammux")) {
        ConfigParser::set_element_properties(streammux_, config.sections.at("streammux"));
    }
    
    // Setup PGIE properties
    if (config.has_section("pgie")) {
        ConfigParser::set_element_properties(pgie_, config.sections.at("pgie"));
    }
    
    // Setup SGIE properties
    if (config.has_section("sgie")) {
        ConfigParser::set_element_properties(sgie_, config.sections.at("sgie"));
    }
    
    // Setup tracker properties
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
    
    // Set default config file if not specified
    std::string config_file = config.get<std::string>("pgie", "config-file-path", "config/config_yolov8n_face.txt");
    g_object_set(G_OBJECT(element),
                 "config-file-path", config_file.c_str(),
                 NULL);
    
    return element;
}

GstElement* Pipeline::create_sgie(const Config& config) {
    GstElement* element = gst_element_factory_make("nvinfer", "sgie");
    if (!element) {
        std::cerr << "Failed to create nvinfer element for sgie" << std::endl;
        return nullptr;
    }
    
    // Set default config file if not specified
    std::string config_file = config.get<std::string>("sgie", "config-file-path", "config/config_arcface.txt");
    g_object_set(G_OBJECT(element),
                 "config-file-path", config_file.c_str(),
                 NULL);
    
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
    
    // Set default properties
    g_object_set(G_OBJECT(element),
                 "rows", 2,
                 "columns", 2,
                 "width", 1280,
                 "height", 720,
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
    
    GstElement* element;
    if (display) {
        // Use display sink
        element = gst_element_factory_make("nveglglessink", "sink");
        if (!element) {
            element = gst_element_factory_make("xvimagesink", "sink");
        }
        if (!element) {
            element = gst_element_factory_make("autovideosink", "sink");
        }
    } else {
        // Use fake sink for no display
        element = gst_element_factory_make("fakesink", "sink");
        if (element) {
            g_object_set(G_OBJECT(element),
                         "sync", 0,
                         "enable-last-sample", 0,
                         NULL);
        }
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