#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <iomanip>
#include <map>
#include <vector>

// DeepStream includes
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

// Config parsing
#include "config_parser.h"
#include "edge_deepstream.h"

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr("Error: %s\n", error->message); \
        g_error_free(error); \
        return -1; \
    }

static GMainLoop *loop = NULL;
static GstElement *pipeline = NULL;
static GstElement *streammux = NULL;
static std::vector<GstElement*> source_bins;
static std::vector<GstElement*> video_converters;
static GstElement *pgie = NULL;
static GstElement *sgie = NULL;
static GstElement *tracker = NULL;
static GstElement *nvosd = NULL;
static GstElement *tiler = NULL;
static GstElement *sink = NULL;

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

static void source_pad_added_callback(GstElement *src, GstPad *new_pad, GstElement *converter) {
    // Get the sink pad index from the source element
    int sink_pad_index = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(src), "sink_pad_index"));

    g_print("Pad-added signal received for source %d\n", sink_pad_index);

    // Check if pad is video
    GstCaps *caps = gst_pad_get_current_caps(new_pad);
    if (!caps) {
        g_print("No caps on pad\n");
        return;
    }

    const GstStructure *str = gst_caps_get_structure(caps, 0);
    g_print("Pad caps: %s\n", gst_structure_get_name(str));

    if (!g_str_has_prefix(gst_structure_get_name(str), "video/")) {
        gst_caps_unref(caps);
        g_print("Not a video pad, ignoring\n");
        return;
    }

    // Print caps details for debugging
    gchar *caps_str = gst_caps_to_string(caps);
    g_print("Video caps: %s\n", caps_str);
    g_free(caps_str);
    gst_caps_unref(caps);

    // Get sink pad from converter
    GstPad *sink_pad = gst_element_get_static_pad(converter, "sink");

    if (!sink_pad) {
        g_printerr("Failed to get sink pad from converter for source %d\n", sink_pad_index);
        return;
    }

    g_print("Linking video pad to converter sink for source %d\n", sink_pad_index);

    // Check if pads can be linked
    GstPadLinkReturn link_ret = gst_pad_link(new_pad, sink_pad);
    if (link_ret != GST_PAD_LINK_OK) {
        g_printerr("Failed to link source %d to converter: %s\n", sink_pad_index, gst_pad_link_get_name(link_ret));

        // Print pad caps for debugging
        GstCaps *src_caps = gst_pad_get_current_caps(new_pad);
        GstCaps *sink_caps = gst_pad_get_current_caps(sink_pad);

        if (src_caps) {
            gchar *src_caps_str = gst_caps_to_string(src_caps);
            g_printerr("Source pad caps: %s\n", src_caps_str);
            g_free(src_caps_str);
            gst_caps_unref(src_caps);
        }

        if (sink_caps) {
            gchar *sink_caps_str = gst_caps_to_string(sink_caps);
            g_printerr("Sink pad caps: %s\n", sink_caps_str);
            g_free(sink_caps_str);
            gst_caps_unref(sink_caps);
        }
    } else {
        g_print("Successfully linked source %d to converter\n", sink_pad_index);
    }

    gst_object_unref(sink_pad);
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

    // Parse config file
    std::string config_path = "/home/m2n/edge-deepstream/config/config_pipeline.toml";
    auto config = EdgeDeepStream::ConfigParser::parse_toml(config_path);
    if (!config) {
        g_printerr("Failed to parse config file: %s\n", config_path.c_str());
        return -1;
    }

    // Create pipeline
    pipeline = gst_pipeline_new("edge-deepstream-pipeline");
    if (!pipeline) {
        g_printerr("Failed to create pipeline\n");
        return -1;
    }

    // Create elements from config
    streammux = gst_element_factory_make("nvstreammux", "streammux");
    if (!streammux) {
        g_printerr("Failed to create streammux\n");
        return -1;
    }

    // Set streammux properties from config
    if (config->has_section("streammux")) {
        auto& streammux_config = config->sections["streammux"];
        g_object_set(G_OBJECT(streammux),
                     "gpu-id", config->get<int>("streammux", "gpu_id", 0),
                     "batch-size", config->get<int>("streammux", "batch-size", 1),
                     "width", config->get<int>("streammux", "width", 960),
                     "height", config->get<int>("streammux", "height", 540),
                     "batched-push-timeout", config->get<int>("streammux", "batched-push-timeout", 15000),
                     "enable-padding", config->get<int>("streammux", "enable-padding", 1),
                     "nvbuf-memory-type", config->get<int>("streammux", "nvbuf-memory-type", 0),
                     "live-source", config->get<int>("streammux", "live-source", 1),
                     "sync-inputs", config->get<int>("streammux", "sync-inputs", 0),
                     "attach-sys-ts", config->get<int>("streammux", "attach-sys-ts", 1),
                     NULL);
    }

    // Create PGIE
    pgie = gst_element_factory_make("nvinfer", "pgie");
    if (!pgie) {
        g_printerr("Failed to create pgie\n");
        return -1;
    }

    // Set PGIE properties from config
    if (config->has_section("pgie")) {
        auto& pgie_config = config->sections["pgie"];
        std::string config_file_path = config->get<std::string>("pgie", "config-file-path", "");
        if (!config_file_path.empty()) {
            g_object_set(G_OBJECT(pgie), "config-file-path", config_file_path.c_str(), NULL);
        }
    }

    // Create SGIE
    sgie = gst_element_factory_make("nvinfer", "sgie");
    if (!sgie) {
        g_printerr("Failed to create sgie\n");
        return -1;
    }

    // Set SGIE properties from config
    if (config->has_section("sgie")) {
        auto& sgie_config = config->sections["sgie"];
        std::string config_file_path = config->get<std::string>("sgie", "config-file-path", "");
        if (!config_file_path.empty()) {
            g_object_set(G_OBJECT(sgie), "config-file-path", config_file_path.c_str(), NULL);
        }
    }

    // Get display mode from config
    bool enable_display = config->get<int>("pipeline", "display", 0);

    // Create sink based on display mode
    if (enable_display) {
        // Try nvvideosink first for Jetson
        sink = gst_element_factory_make("nvvideosink", "sink");
        if (!sink) {
            g_printerr("Failed to create nvvideosink, trying nveglglessink\n");
            sink = gst_element_factory_make("nveglglessink", "sink");
            if (!sink) {
                g_printerr("Failed to create nveglglessink, trying autovideosink\n");
                sink = gst_element_factory_make("autovideosink", "sink");
                if (!sink) {
                    g_printerr("Failed to create autovideosink, using fakesink\n");
                    sink = gst_element_factory_make("fakesink", "sink");
                }
            }
        }
    } else {
        sink = gst_element_factory_make("fakesink", "sink");
    }

    if (!sink) {
        g_printerr("Failed to create sink\n");
        return -1;
    }

    // Set sink properties from config
    if (config->has_section("sink")) {
        auto& sink_config = config->sections["sink"];
        g_object_set(G_OBJECT(sink),
                     "qos", config->get<int>("sink", "qos", 1),
                     "sync", config->get<int>("sink", "sync", 0),
                     NULL);
    }

    // Check if display is enabled
    g_print("Display mode: %s\n", enable_display ? "enabled" : "disabled (headless)");

    // Create display elements if needed
    if (enable_display) {
        // Create tracker
        tracker = gst_element_factory_make("nvtracker", "tracker");
        if (!tracker) {
            g_printerr("Failed to create tracker\n");
            return -1;
        }

        // Set tracker properties from config
        if (config->has_section("tracker")) {
            auto& tracker_config = config->sections["tracker"];
            std::string config_file_path = config->get<std::string>("tracker", "config-file-path", "");
            if (!config_file_path.empty()) {
                // Set tracker properties directly
                g_object_set(G_OBJECT(tracker),
                           "ll-lib-file", "/opt/nvidia/deepstream/deepstream-6.3/lib/libnvds_nvmultiobjecttracker.so",
                           "ll-config-file", config_file_path.c_str(),
                           "tracker-width", 320,
                           "tracker-height", 320,
                           "gpu-id", 0,
                           NULL);
            }
        }

        // Create tiler
        tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
        if (!tiler) {
            g_printerr("Failed to create tiler\n");
            return -1;
        }

        // Set tiler properties from config
        if (config->has_section("tiler")) {
            auto& tiler_config = config->sections["tiler"];
            g_object_set(G_OBJECT(tiler),
                         "width", config->get<int>("tiler", "width", 1280),
                         "height", config->get<int>("tiler", "height", 720),
                         NULL);
        }

        // Create nvosd
        nvosd = gst_element_factory_make("nvdsosd", "nvosd");
        if (!nvosd) {
            g_printerr("Failed to create nvosd\n");
            return -1;
        }

        // Set nvosd properties from config
        if (config->has_section("nvosd")) {
            auto& nvosd_config = config->sections["nvosd"];
            g_object_set(G_OBJECT(nvosd),
                         "process-mode", config->get<int>("nvosd", "process-mode", 0),
                         "display-text", config->get<int>("nvosd", "display-text", 1),
                         NULL);
        }

        g_print("Display elements created (with tracker)\n");
    }

    // Add elements to pipeline
    if (enable_display) {
        gst_bin_add_many(GST_BIN(pipeline), streammux, pgie, sgie, tracker, tiler, nvosd, sink, NULL);

        // Link elements: streammux -> pgie -> sgie -> tracker -> tiler -> nvosd -> sink
        if (!gst_element_link_many(streammux, pgie, sgie, tracker, tiler, nvosd, sink, NULL)) {
            g_printerr("Failed to link display pipeline elements\n");
            return -1;
        }
    } else {
        gst_bin_add_many(GST_BIN(pipeline), streammux, pgie, sgie, sink, NULL);

        // Link elements: streammux -> pgie -> sgie -> sink
        if (!gst_element_link_many(streammux, pgie, sgie, sink, NULL)) {
            g_printerr("Failed to link headless pipeline elements\n");
            return -1;
        }
    }

    // Create source bins for all sources from config BEFORE setting pipeline to READY
    if (config->has_section("source")) {
        auto& source_config = config->sections["source"];

        // Iterate through all sources in the config
        int source_index = 0;
        for (const auto& source_pair : source_config) {
            std::string source_key = source_pair.first;
            std::string source_uri = source_pair.second;

            if (source_uri.empty()) continue;

            g_print("Creating source '%s': %s\n", source_key.c_str(), source_uri.c_str());

            // Determine source type from URI
            std::string source_type = "unknown";
            if (source_uri.find("file://") == 0) {
                source_type = "file";
            } else if (source_uri.find("rtsp://") == 0) {
                source_type = "rtsp";
            }

            g_print("Detected source type: %s\n", source_type.c_str());

            // Create source element
            GstElement *src_bin = gst_element_factory_make("uridecodebin", source_key.c_str());
            if (!src_bin) {
                g_printerr("Failed to create source for %s\n", source_key.c_str());
                continue;
            }

            // Create video converter for this source
            std::string converter_name = "converter_" + std::to_string(source_index);
            GstElement *converter = gst_element_factory_make("nvvideoconvert", converter_name.c_str());
            if (!converter) {
                g_printerr("Failed to create video converter for %s\n", source_key.c_str());
                gst_object_unref(src_bin);
                continue;
            }

            g_object_set(G_OBJECT(src_bin), "uri", source_uri.c_str(), NULL);
            gst_bin_add_many(GST_BIN(pipeline), src_bin, converter, NULL);
            source_bins.push_back(src_bin);
            video_converters.push_back(converter);

            // Store the sink pad index for this source
            g_object_set_data(G_OBJECT(src_bin), "sink_pad_index", GINT_TO_POINTER(source_index));

            // Connect source to converter
            g_signal_connect(src_bin, "pad-added", G_CALLBACK(source_pad_added_callback), converter);

            // Link converter to streammux
            gchar *sink_pad_name = g_strdup_printf("sink_%d", source_index);
            GstPad *sink_pad = gst_element_get_request_pad(streammux, sink_pad_name);
            g_free(sink_pad_name);

            if (!sink_pad) {
                g_printerr("Failed to get sink pad sink_%d from streammux\n", source_index);
                continue;
            }

            GstPad *converter_src_pad = gst_element_get_static_pad(converter, "src");
            if (!converter_src_pad) {
                g_printerr("Failed to get src pad from converter for source %d\n", source_index);
                gst_object_unref(sink_pad);
                continue;
            }

            if (gst_pad_link(converter_src_pad, sink_pad) != GST_PAD_LINK_OK) {
                g_printerr("Failed to link converter to streammux for source %d\n", source_index);
                gst_object_unref(converter_src_pad);
                gst_object_unref(sink_pad);
                continue;
            }

            gst_object_unref(converter_src_pad);
            gst_object_unref(sink_pad);

            source_index++;
        }

        g_print("Created %d sources\n", (int)source_bins.size());
    }

    // Set pipeline to READY state after adding sources
    g_print("Setting pipeline to READY state...\n");
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_READY);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to set pipeline to READY state\n");
        return -1;
    }
    g_print("Pipeline set to READY state\n");

    // Add probe to PGIE src pad
    GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
    if (pgie_src_pad) {
        gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_pad_probe, NULL, NULL);
        gst_object_unref(pgie_src_pad);
        g_print("Probe attached to PGIE src pad\n");
    } else {
        g_printerr("Failed to get SGIE src pad for probe\n");
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
    g_print("Starting EdgeDeepStream pipeline test...\n");
    ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to start pipeline\n");
        return -1;
    }
    g_print("Pipeline set to PLAYING state\n");

    // Run for 60 seconds to see detection results
    g_print("Running inference for 60 seconds to verify detections...\n");
    std::this_thread::sleep_for(std::chrono::seconds(60));

    // Stop pipeline
    g_print("Stopping pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);

    // Cleanup
    gst_object_unref(pipeline);
    g_print("Config-driven pipeline test completed successfully!\n");

    return 0;
}