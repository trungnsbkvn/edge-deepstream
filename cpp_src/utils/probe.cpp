#include <gst/gst.h>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

namespace EdgeDeepStream {

// Forward declarations
struct ProbeContext;

/**
 * PGIE (YOLOv8n face detection) source pad probe
 * Extracts face detection metadata after face detection inference
 */
GstPadProbeReturn pgie_src_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    static int frame_count = 0;
    static std::unordered_map<int, int> stream_frame_counts;
    frame_count++;
    
    // Add periodic debug output
    if (frame_count % 30 == 1) {
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
        if (stream_frame_counts[stream_id] % 30 == 1) {
            g_print("PGIE: Stream %d - Received %d frames (RTSP stream active)\n", 
                    stream_id, stream_frame_counts[stream_id]);
        }
        
        // Process object metadata from PGIE
        NvDsMetaList* l_obj = frame_meta->obj_meta_list;
        int obj_count = 0;
        while (l_obj != NULL) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)(l_obj->data);
            obj_count++;
            
            // Filter objects by confidence (similar to Python version)
            if (obj_meta->confidence > 0.6f && frame_count % 30 == 1) {
                g_print("PGIE: Frame %d - Detected object class_id=%d confidence=%.2f\n", 
                        frame_count, obj_meta->class_id, obj_meta->confidence);
            }
            
            l_obj = l_obj->next;
        }
        
        if (frame_count % 30 == 1 && obj_count > 0) {
            g_print("PGIE: Frame %d - Total objects: %d\n", frame_count, obj_count);
        }
        
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
    
    if (!debug_logged) {
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
        
        // Count objects with user metadata (feature vectors)
        int feature_count = 0;
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)(l_obj->data);
            
            // Check if this object has user metadata (feature vectors from SGIE)
            for (NvDsMetaList* l_user = obj_meta->obj_user_meta_list; l_user != nullptr; l_user = l_user->next) {
                NvDsUserMeta* user_meta = (NvDsUserMeta*)(l_user->data);
                if (user_meta && user_meta->user_meta_data) {
                    feature_count++;
                    
                    // Here you would normally:
                    // 1. Extract the feature vector from user_meta->user_meta_data
                    // 2. Compare against known faces database
                    // 3. Update obj_meta with recognition results
                    // For now, just log that we found features
                    break;
                }
            }
        }
        
        if (feature_count > 0) {
            g_print("[SGIE_PROBE] Frame %u: Extracted %d feature vectors\n", frame_meta->frame_num, feature_count);
        }
    }

    return GST_PAD_PROBE_OK;
}

/**
 * Attach probes to pipeline elements
 */
bool attach_probes(GstElement* pgie, GstElement* sgie, void* probe_context) {
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

    if (sgie) {
        g_print("Successfully attached probes: PGIE probe_id=%lu, SGIE probe_id=%lu\n", 
                pgie_probe_id, sgie_probe_id);
    } else {
        g_print("Successfully attached probes: PGIE probe_id=%lu (no SGIE)\n", pgie_probe_id);
    }
    return true;
}

} // namespace EdgeDeepStream