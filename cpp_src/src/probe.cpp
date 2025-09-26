#include "probe.h"
#include <iostream>
#include <iomanip>
#include <atomic>

namespace EdgeDeepStream {

// Global counters for PGIE probe statistics
static std::atomic<int> frame_count{0};
static std::atomic<int> total_detections{0};

/**
 * PGIE (YOLOv8n face detection) source pad probe function
 * Extracts face detection metadata after face detection inference
 */
GstPadProbeReturn pgie_src_filter_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    frame_count++;

    if (GST_IS_BUFFER(info->data)) {
        GstBuffer *buf = GST_BUFFER(info->data);
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

        if (batch_meta) {
            int batch_detections = 0;
            for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
                NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

                batch_detections += frame_meta->num_obj_meta;

                if (frame_meta->num_obj_meta > 0) {
                    std::cout << "Frame " << frame_count << ": " << frame_meta->num_obj_meta << " faces detected" << std::endl;

                    for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
                        std::cout << "  Face: confidence=" << std::fixed << std::setprecision(3) << obj_meta->confidence
                                 << " bbox=(" << obj_meta->rect_params.left << ","
                                 << obj_meta->rect_params.top << ","
                                 << obj_meta->rect_params.width << ","
                                 << obj_meta->rect_params.height << ")" << std::endl;
                    }
                }
            }

            total_detections += batch_detections;

            // Print stats every 30 frames
            if (frame_count % 30 == 0) {
                std::cout << "=== Stats: Frame " << frame_count << ", Total detections: " << total_detections
                         << ", Avg: " << std::fixed << std::setprecision(2) << (float)total_detections/frame_count << " faces/frame ===" << std::endl;
            }
        } else {
            // Debug: print when we get buffers but no batch meta
            if (frame_count % 100 == 0) {
                std::cout << "[DEBUG] Frame " << frame_count << ": Buffer received but no batch meta" << std::endl;
            }
        }
    } else {
        std::cout << "[DEBUG] Non-buffer data received in PGIE probe" << std::endl;
    }

    return GST_PAD_PROBE_OK;
}

/**
 * SGIE (ArcFace feature extraction) source pad probe function
 * Extracts face feature vectors after feature extraction inference
 */
GstPadProbeReturn sgie_feature_extract_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    // TODO: Implement SGIE probe for feature extraction
    // This will extract face features and perform recognition
    return GST_PAD_PROBE_OK;
}

/**
 * Attach probe functions to pipeline elements
 * @param pgie Primary inference engine (YOLOv8n face detection)
 * @param sgie Secondary inference engine (ArcFace feature extraction)
 * @param probe_context Optional context data for probes
 * @return true if probes attached successfully, false otherwise
 */
bool attach_probes(GstElement* pgie, GstElement* sgie, void* probe_context) {
    bool success = true;

    // Attach PGIE probe to src pad
    if (pgie) {
        GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
        if (pgie_src_pad) {
            gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                            pgie_src_filter_probe, probe_context, NULL);
            gst_object_unref(pgie_src_pad);
            std::cout << "[PROBE] PGIE probe attached successfully" << std::endl;
        } else {
            std::cerr << "[PROBE] Failed to get PGIE src pad for probe attachment" << std::endl;
            success = false;
        }
    }

    // TODO: Attach SGIE probe when SGIE is implemented
    if (sgie) {
        // SGIE probe attachment will be added here when SGIE is integrated
    }

    return success;
}

} // namespace EdgeDeepStream