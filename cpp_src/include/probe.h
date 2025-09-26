#pragma once

#include <gst/gst.h>
#include "gstnvdsmeta.h"

namespace EdgeDeepStream {

/**
 * PGIE (YOLOv8n face detection) source pad probe function
 * Extracts face detection metadata after face detection inference
 */
GstPadProbeReturn pgie_src_filter_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data);

/**
 * SGIE (ArcFace feature extraction) source pad probe function  
 * Extracts face feature vectors after feature extraction inference
 */
GstPadProbeReturn sgie_feature_extract_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data);

/**
 * Attach probe functions to pipeline elements
 * @param pgie Primary inference engine (YOLOv8n face detection)
 * @param sgie Secondary inference engine (ArcFace feature extraction) 
 * @param probe_context Optional context data for probes
 * @return true if probes attached successfully, false otherwise
 */
bool attach_probes(GstElement* pgie, GstElement* sgie, void* probe_context = nullptr);

} // namespace EdgeDeepStream