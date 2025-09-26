#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <iomanip>

// DeepStream includes
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr("Error: %s\n", error->message); \
        g_error_free(error); \
        return -1; \
    }

static GMainLoop *loop = NULL;
static GstElement *pipeline = NULL;
static GstElement *streammux = NULL;
static GstElement *pgie = NULL;
static GstElement *sgie = NULL;
static GstElement *fakesink = NULL;

static GstPadProbeReturn pgie_pad_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static int frame_count = 0;
    static int total_detections = 0;
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
        }
    }

    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn sgie_pad_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static int sgie_frame_count = 0;
    sgie_frame_count++;

    if (GST_IS_BUFFER(info->data)) {
        GstBuffer *buf = GST_BUFFER(info->data);
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

        if (batch_meta) {
            for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
                NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

                if (frame_meta->num_obj_meta > 0) {
                    std::cout << "SGIE Frame " << sgie_frame_count << ": " << frame_meta->num_obj_meta << " faces processed by ArcFace" << std::endl;

                    for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

                        std::cout << "  Refined Face: confidence=" << std::fixed << std::setprecision(3) << obj_meta->confidence
                                 << " bbox=(" << obj_meta->rect_params.left << ","
                                 << obj_meta->rect_params.top << ","
                                 << obj_meta->rect_params.width << ","
                                 << obj_meta->rect_params.height << ")" << std::endl;

                        // Check for classifier meta (landmarks might be stored here)
                        for (NvDsMetaList *l_classifier = obj_meta->classifier_meta_list; l_classifier != NULL; l_classifier = l_classifier->next) {
                            NvDsClassifierMeta *classifier_meta = (NvDsClassifierMeta *)l_classifier->data;
                            std::cout << "    Classifier: " << classifier_meta->classifier_type << std::endl;
                        }
                    }
                }
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            g_printerr("Debug info: %s\n", debug ? debug : "none");
            g_error_free(error);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED: {
            GstState old_state, new_state;
            gst_message_parse_state_changed(msg, &old_state, &new_state, NULL);
            if (GST_OBJECT_NAME(msg->src) == std::string("pipeline")) {
                g_print("Pipeline state changed from %s to %s\n",
                       gst_element_state_get_name(old_state),
                       gst_element_state_get_name(new_state));
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // Create pipeline
    pipeline = gst_pipeline_new("yolov8n-test-pipeline");
    if (!pipeline) {
        g_printerr("Failed to create pipeline\n");
        return -1;
    }

    // Create elements
    streammux = gst_element_factory_make("nvstreammux", "streammux");
    pgie = gst_element_factory_make("nvinfer", "pgie");
    sgie = gst_element_factory_make("nvinfer", "sgie");
    fakesink = gst_element_factory_make("fakesink", "fakesink");

    if (!streammux || !pgie || !sgie || !fakesink) {
        g_printerr("Failed to create elements\n");
        return -1;
    }

    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline), streammux, pgie, sgie, fakesink, NULL);

    // Set properties
    g_object_set(G_OBJECT(streammux),
                 "batch-size", 1,
                 "width", 960,
                 "height", 540,
                 "batched-push-timeout", 15000,
                 "enable-padding", 1,
                 "nvbuf-memory-type", 0,
                 "live-source", 1,
                 "sync-inputs", 0,
                 "attach-sys-ts", 1,
                 NULL);

    g_object_set(G_OBJECT(pgie),
                 "config-file-path", "/home/m2n/edge-deepstream/config/config_yolov8n_face.txt",
                 NULL);

    g_object_set(G_OBJECT(sgie),
                 "config-file-path", "/home/m2n/edge-deepstream/config/config_arcface.txt",
                 NULL);

    g_object_set(G_OBJECT(fakesink),
                 "sync", 0,
                 "qos", 1,
                 NULL);

    // Link elements: streammux -> pgie -> sgie -> fakesink
    if (!gst_element_link_many(streammux, pgie, sgie, fakesink, NULL)) {
        g_printerr("Failed to link elements\n");
        return -1;
    }

    // Create source bin for video file
    GstElement *source_bin = gst_element_factory_make("uridecodebin", "source");
    if (!source_bin) {
        g_printerr("Failed to create source\n");
        return -1;
    }

    g_object_set(G_OBJECT(source_bin),
                 "uri", "file:///home/m2n/edge-deepstream/data/media/friends_s1e1_cut.mp4",
                 NULL);

    gst_bin_add(GST_BIN(pipeline), source_bin);

    // Connect source to streammux
    g_signal_connect(source_bin, "pad-added", G_CALLBACK(+[](GstElement *src, GstPad *new_pad, GstElement *mux) {
        // Check if pad is video
        GstCaps *caps = gst_pad_get_current_caps(new_pad);
        if (!caps) return;

        const GstStructure *str = gst_caps_get_structure(caps, 0);
        if (!g_str_has_prefix(gst_structure_get_name(str), "video/")) {
            gst_caps_unref(caps);
            return;
        }
        gst_caps_unref(caps);

        // Get sink pad from streammux
        GstPad *sink_pad = gst_element_get_request_pad(mux, "sink_0");
        if (!sink_pad) {
            g_printerr("Failed to get sink pad from streammux\n");
            return;
        }

        g_print("Linking video pad to streammux sink_0\n");
        if (gst_pad_link(new_pad, sink_pad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link source to mux\n");
        }
        gst_object_unref(sink_pad);
    }), streammux);

        // Add probe to PGIE src pad
    GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
    if (pgie_src_pad) {
        gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_pad_probe, NULL, NULL);
        gst_object_unref(pgie_src_pad);
        g_print("Probe attached to PGIE src pad\n");
    } else {
        g_printerr("Failed to get PGIE src pad for probe\n");
    }

    // Add probe to SGIE src pad
    GstPad *sgie_src_pad = gst_element_get_static_pad(sgie, "src");
    if (sgie_src_pad) {
        gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_pad_probe, NULL, NULL);
        gst_object_unref(sgie_src_pad);
        g_print("Probe attached to SGIE src pad\n");
    } else {
        g_printerr("Failed to get SGIE src pad for probe\n");
    }

    // Add bus watch
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, NULL);
    gst_object_unref(bus);

    // Set pipeline to playing
    g_print("Starting YOLOv8n model test...\n");
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to start pipeline\n");
        return -1;
    }
    g_print("Pipeline set to PLAYING state\n");

    // Run for 30 seconds to see detection results
    g_print("Running inference for 30 seconds to verify detections...\n");
    std::this_thread::sleep_for(std::chrono::seconds(30));

    // Stop pipeline
    g_print("Stopping pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);

    // Cleanup
    gst_object_unref(pipeline);
    g_print("Test completed successfully!\n");

    return 0;
}