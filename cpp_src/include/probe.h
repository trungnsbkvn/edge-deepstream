#pragma once

#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvds_meta.h>
#include <string>
#include <memory>
#include <vector>

// Forward declarations
namespace EdgeDeepStream {
    class FaissIndex;
    class TensorRTInfer;
    class EventSender;
    class PerfStats;
}

namespace EdgeDeepStream {
namespace Probe {

/**
 * Structure to hold probe configuration and runtime data
 */
struct ProbeData {
    // Pipeline configuration
    std::string config_path;
    double pgie_min_conf = 0.45;
    double recognition_threshold = 0.5;
    bool save_recognized = true;
    std::string save_dir;
    std::string save_mode = "first";  // "all", "first", "best"
    
    // Recognition configuration
    bool use_index = true;
    std::string metric = "cosine";
    std::string backend = "faiss";
    bool recognize_once_per_track = true;
    
    // Cache settings
    int max_track_embeddings = 30;
    int min_embeddings_for_fusion = 1;
    std::string fusion_mode = "mean";
    
    // Event settings
    bool events_enabled = false;
    std::string unix_socket_path;
    bool send_image = true;
    
    // Debug settings
    bool verbose = false;
    int perf_verbose = 0;
    
    // Runtime components
    std::shared_ptr<FaissIndex> faiss_index;
    std::shared_ptr<TensorRTInfer> tensorrt_engine;
    std::shared_ptr<EventSender> event_sender;
    
    // Track state management
    struct TrackState {
        std::vector<std::vector<float>> embeddings;
        std::string best_label;
        double best_score = 0.0;
        bool recognized = false;
        int frame_count = 0;
    };
    std::unordered_map<uint64_t, TrackState> track_cache;
    
    // Display name mapping
    std::unordered_map<std::string, std::string> display_names;
    std::string labels_path;
};

/**
 * Initialize probe data structure from configuration
 */
bool initialize_probe_data(ProbeData* data, const std::string& config_path);

/**
 * Primary inference probe - processes face detection results
 * Attaches to PGIE (primary inference engine) output
 */
GstPadProbeReturn pgie_src_filter_probe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);

/**
 * Secondary inference probe - processes face recognition 
 * Attaches to SGIE (secondary inference engine) output
 */
GstPadProbeReturn sgie_feature_extract_probe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);

/**
 * Extract face feature from object metadata
 */
bool get_face_feature(NvDsObjectMeta* obj_meta, guint frame_num, ProbeData* data, 
                      std::vector<float>& feature);

/**
 * Perform face recognition using FAISS index
 */
struct RecognitionResult {
    std::string label;
    double score = 0.0;
    bool matched = false;
};

RecognitionResult recognize_face(const std::vector<float>& feature, ProbeData* data);

/**
 * Manage track-level embeddings and fusion
 */
void update_track_embeddings(uint64_t track_id, const std::vector<float>& embedding, 
                           ProbeData* data);

/**
 * Get fused embedding for a track
 */
std::vector<float> get_fused_embedding(uint64_t track_id, ProbeData* data);

/**
 * Send recognition event
 */
void send_recognition_event(const RecognitionResult& result, NvDsObjectMeta* obj_meta,
                          guint frame_num, ProbeData* data);

/**
 * Save recognized face image
 */
bool save_face_crop(NvDsObjectMeta* obj_meta, guint frame_num, 
                   const RecognitionResult& result, ProbeData* data);

/**
 * Clear recognition caches for specific user
 */
void clear_name_cache_for_user(const std::string& user_id);

/**
 * Clear all recognition caches
 */
void clear_all_recognition_caches();

/**
 * Invalidate user from active track cache
 */
void invalidate_user_from_active_cache(const std::string& user_id, ProbeData* data);

/**
 * Get display name from label
 */
std::string get_display_name(const std::string& label, ProbeData* data);

/**
 * Update performance statistics
 */
void update_perf_stats(const std::string& event, double duration_ms = 0.0);

} // namespace Probe
} // namespace EdgeDeepStream