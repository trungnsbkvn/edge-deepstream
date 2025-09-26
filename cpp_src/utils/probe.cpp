#include <gst/gst.h>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>
#include <functional>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

namespace EdgeDeepStream {

// Forward declarations and callback types
struct ProbeContext;

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
 * Global verbose flag for debug output
 */
static bool g_verbose = false;

/**
 * PGIE (YOLOv8n face detection) source pad probe
 * Extracts face detection metadata after face detection inference
 */
GstPadProbeReturn pgie_src_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    static int frame_count = 0;
    static std::unordered_map<int, int> stream_frame_counts;
    frame_count++;
    
    // Add periodic debug output
    if (g_verbose && frame_count % 30 == 1) {
        g_print("PGIE probe: Processing frame %d (total frames received)\n", frame_count);
    }

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) {
        return GST_PAD_PROBE_OK;
    }

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    NvDsMetaList* l_frame = batch_meta->frame_meta_list;
    while (l_frame != NULL) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
        
        // Track per-stream frame counts
        int stream_id = frame_meta->source_id;
        stream_frame_counts[stream_id]++;
        
        // Print RTSP stream reception status
        if (g_verbose && stream_frame_counts[stream_id] % 30 == 1) {
            g_print("PGIE: Stream %d - Received %d frames (RTSP stream active)\n", 
                    stream_id, stream_frame_counts[stream_id]);
        }
        
        // Collect detected faces for callback
        std::vector<NvDsObjectMeta*> detected_faces;
        
        // Process object metadata from PGIE
        NvDsMetaList* l_obj = frame_meta->obj_meta_list;
        int obj_count = 0;
        while (l_obj != NULL) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)(l_obj->data);
            obj_count++;
            
            // Filter objects by confidence threshold (using default 0.6)
            if (obj_meta->confidence >= 0.6f) {
                detected_faces.push_back(obj_meta);
                
                if (g_verbose) {
                    g_print("PGIE: Frame %d - Face detected: class_id=%d confidence=%.3f\n", 
                            frame_count, obj_meta->class_id, obj_meta->confidence);
                    g_print("      Box: (%.1f, %.1f, %.1f, %.1f)\n", 
                            obj_meta->rect_params.left, obj_meta->rect_params.top,
                            obj_meta->rect_params.width, obj_meta->rect_params.height);
                    
                    // Print landmarks if available
                    if (obj_meta->mask_params.data) {
                        // YOLOv8n face outputs 5 landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
                        float* landmarks = (float*)obj_meta->mask_params.data;
                        g_print("      Landmarks: [");
                        for (int i = 0; i < 10; i += 2) {  // 5 points * 2 coordinates
                            g_print("(%.1f,%.1f)", landmarks[i], landmarks[i+1]);
                            if (i < 8) g_print(", ");
                        }
                        g_print("]\n");
                    }
                }
            }
            
            l_obj = l_obj->next;
        }
        
        if (g_verbose && frame_count % 30 == 1 && obj_count > 0) {
            g_print("PGIE: Frame %d - Total objects: %d, Faces above threshold: %zu\n", 
                    frame_count, obj_count, detected_faces.size());
        }
        
        // Call face detection callback if registered
        // if (g_probe_config.on_face_detected && !detected_faces.empty()) {
        //     g_probe_config.on_face_detected(frame_meta->frame_num, stream_id, detected_faces);
        // }
        
        l_frame = l_frame->next;
    }

    return GST_PAD_PROBE_OK;
}

/**
 * SGIE (ArcFace feature extraction) source pad probe
 * Extracts face feature vectors after feature extraction inference
 */
GstPadProbeReturn sgie_feature_extract_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    static bool debug_logged = false;
    
    if (g_verbose && !debug_logged) {
        g_print("[SGIE_PROBE] Attached and processing buffers\n");
        debug_logged = true;
    }

    GstBuffer* gst_buffer = gst_pad_probe_info_get_buffer(info);
    if (!gst_buffer) {
        g_print("Unable to get GstBuffer\n");
        return GST_PAD_PROBE_OK;
    }

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    // Iterate through each frame in the batch
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
        int stream_id = frame_meta->source_id;
        
        // Count objects with user metadata (feature vectors)
        int feature_count = 0;
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)(l_obj->data);
            
            // Check if this object has user metadata (feature vectors from SGIE)
            for (NvDsMetaList* l_user = obj_meta->obj_user_meta_list; l_user != nullptr; l_user = l_user->next) {
                NvDsUserMeta* user_meta = (NvDsUserMeta*)(l_user->data);
                if (user_meta && user_meta->user_meta_data) {
                    feature_count++;
                    
                    // Extract feature vector from user metadata
                    // Here you would normally:
                    // 1. Cast user_meta->user_meta_data to the appropriate structure
                    // 2. Extract the feature vector (e.g., NvDsInferTensorMeta)
                    // 3. Convert to std::vector<float> for further processing
                    
                    std::vector<float> features;
                    // TODO: Implement actual feature extraction from user_meta->user_meta_data
                    
                    // Call feature extraction callback if registered
                    // if (g_probe_config.on_features_extracted) {
                    //     g_probe_config.on_features_extracted(frame_meta->frame_num, stream_id, obj_meta, features);
                    // }
                    
                    // Here you would normally:
                    // 4. Compare features against known faces database
                    // 5. Perform face recognition
                    // 6. Update obj_meta with recognition results
                    // 7. Call face recognition callback
                    
                    // Example recognition logic (to be implemented):
                    // std::string identity = "Unknown";
                    // float recognition_confidence = 0.0f;
                    // TODO: Implement face recognition logic
                    // if (g_probe_config.on_face_recognized) {
                    //     g_probe_config.on_face_recognized(frame_meta->frame_num, stream_id, obj_meta, identity, recognition_confidence);
                    // }
                    
                    break; // Assuming one feature vector per object
                }
            }
        }
        
        if (g_verbose && feature_count > 0) {
            g_print("[SGIE_PROBE] Frame %u: Extracted %d feature vectors\n", frame_meta->frame_num, feature_count);
        }
    }

    return GST_PAD_PROBE_OK;
}

/**
 * Set global probe configuration
 * Allows customization of probe behavior through callbacks
 */
// void set_probe_config(const ProbeConfig& config) {
//     g_probe_config = config;
// }

/**
 * Get current probe configuration
 */
// const ProbeConfig& get_probe_config() {
//     return g_probe_config;
// }

/**
 * Attach probes to pipeline elements
 */
bool attach_probes(GstElement* pgie, GstElement* sgie, void* probe_context, bool verbose) {
    g_verbose = verbose;
    
    if (!pgie) {
        g_print("ERROR: Cannot attach probes - PGIE is null\n");
        return false;
    }

    // Attach probe to PGIE source pad
    GstPad* pgie_src_pad = gst_element_get_static_pad(pgie, "src");
    if (!pgie_src_pad) {
        g_print("ERROR: Failed to get PGIE source pad\n");
        return false;
    }

    gulong pgie_probe_id = gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                                             pgie_src_pad_buffer_probe, probe_context, nullptr);
    gst_object_unref(pgie_src_pad);
    
    if (pgie_probe_id == 0) {
        g_print("ERROR: Failed to attach PGIE probe\n");
        return false;
    }

    // Attach probe to SGIE source pad (if SGIE exists)
    gulong sgie_probe_id = 1; // Default non-zero value
    if (sgie) {
        GstPad* sgie_src_pad = gst_element_get_static_pad(sgie, "src");
        if (!sgie_src_pad) {
            g_print("ERROR: Failed to get SGIE source pad\n");
            return false;
        }

        sgie_probe_id = gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                                                 sgie_feature_extract_probe, probe_context, nullptr);
        gst_object_unref(sgie_src_pad);
    }
    
    if (sgie && sgie_probe_id == 0) {
        g_print("ERROR: Failed to attach SGIE probe\n");
        return false;
    }

    if (g_verbose) {
        if (sgie) {
            g_print("Successfully attached probes: PGIE probe_id=%lu, SGIE probe_id=%lu\n", 
                    pgie_probe_id, sgie_probe_id);
        } else {
            g_print("Successfully attached probes: PGIE probe_id=%lu (no SGIE)\n", pgie_probe_id);
        }
    }
    
    return true;
}

} // namespace EdgeDeepStream