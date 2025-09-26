#pragma once

#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include <functional>
#include <vector>
#include <string>

namespace EdgeDeepStream {

/**
 * Callback function type for face detection results
 * Called when faces are detected in PGIE output
 */
using FaceDetectionCallback = std::function<void(
    int frame_num, 
    int stream_id, 
    const std::vector<NvDsObjectMeta*>& detected_faces
)>;

/**
 * Callback function type for feature extraction results
 * Called when features are extracted from SGIE output
 */
using FeatureExtractionCallback = std::function<void(
    int frame_num,
    int stream_id,
    NvDsObjectMeta* face_obj,
    const std::vector<float>& features
)>;

/**
 * Callback function type for face recognition results
 * Called when face recognition is performed
 */
using FaceRecognitionCallback = std::function<void(
    int frame_num,
    int stream_id,
    NvDsObjectMeta* face_obj,
    const std::string& recognized_identity,
    float confidence
)>;

/**
 * Probe configuration structure
 * Allows customization of probe behavior through callbacks
 */
struct ProbeConfig {
    FaceDetectionCallback on_face_detected = nullptr;
    FeatureExtractionCallback on_features_extracted = nullptr;
    FaceRecognitionCallback on_face_recognized = nullptr;
    
    // Enable/disable debug logging
    bool enable_debug_logging = true;
    
    // Minimum confidence threshold for face detection
    float face_detection_threshold = 0.6f;
};

/**
 * PGIE (YOLOv8n face detection) source pad probe function
 * Extracts face detection metadata after face detection inference
 */
GstPadProbeReturn pgie_src_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data);

/**
 * SGIE (ArcFace feature extraction) source pad probe function  
 * Extracts face feature vectors after feature extraction inference
 */
GstPadProbeReturn sgie_feature_extract_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data);

/**
 * Set global probe configuration
 * Allows customization of probe behavior through callbacks
 */
// void set_probe_config(const ProbeConfig& config);

/**
 * Get current probe configuration
 */
// const ProbeConfig& get_probe_config();

/**
 * Attach probe functions to pipeline elements
 * @param pgie Primary inference engine (YOLOv8n face detection)
 * @param sgie Secondary inference engine (ArcFace feature extraction) 
 * @param probe_context Optional context data for probes
 * @param verbose Enable verbose debug output
 * @return true if probes attached successfully, false otherwise
 */
bool attach_probes(GstElement* pgie, GstElement* sgie, void* probe_context = nullptr, bool verbose = false);

} // namespace EdgeDeepStream