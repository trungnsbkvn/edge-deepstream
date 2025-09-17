#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import sys
import os
import math
import time

from utils.probe import *
from utils.parser_cfg import *
from utils.bus_call import bus_call

# Global realtime flag to allow decoder frame dropping
REALTIME_DROP = False


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")
    elif (gstname.find("audio") != -1):
        # Consume audio by linking to a fakesink to avoid audio warnings
        try:
            fake = Gst.ElementFactory.make("fakesink", None)
            if fake:
                fake.set_property('enable-last-sample', 0)
                fake.set_property('sync', 0)
                source_bin.add(fake)
                fake.sync_state_with_parent()
                sinkpad = fake.get_static_pad("sink")
                if decoder_src_pad.can_link(sinkpad):
                    decoder_src_pad.link(sinkpad)
                else:
                    source_bin.remove(fake)
        except Exception:
            pass


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name)
    if (name.find("decodebin") != -1):
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)
    # Configure NVIDIA decoder for realtime dropping if enabled
    try:
        if REALTIME_DROP and ("nvv4l2decoder" in name or "nvh264dec" in name or "nvh265dec" in name):
            if Object.find_property('drop-frame-interval') is not None:
                Object.set_property('drop-frame-interval', 1)
            if Object.find_property('disable-dpb') is not None:
                Object.set_property('disable-dpb', True)
    except Exception:
        pass


def create_source_bin(index, uri):
    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Prefer NVMM video surfaces and avoid exposing non-video streams to keep pipeline lean
    try:
        uri_decode_bin.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        # Some versions support expose-all-streams: try to restrict to video only
        if uri_decode_bin.find_property('expose-all-streams') is not None:
            uri_decode_bin.set_property('expose-all-streams', False)
    except Exception:
        pass
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(cfg, run_duration=None):
    start_wall = time.time()
    global REALTIME_DROP
    print(cfg)
    print("Load known faces features \n")
    known_face_features = load_faces(cfg['pipeline']['known_face_dir'])
    # Load prebuilt vector index (FAISS) if available and enabled (no building at runtime)
    try:
        recog_cfg = cfg.get('recognition', {})
    except Exception:
        recog_cfg = {}
    try:
        vector_index = safe_load_index(recog_cfg)
        if vector_index is not None:
            try:
                print(f"Vector index ready: {vector_index.size()} entries\n")
            except Exception:
                print("Vector index loaded.\n")
        else:
            reason = []
            try:
                if os.getenv('DS_DISABLE_FAISS', '0') == '1':
                    reason.append('env DS_DISABLE_FAISS=1')
                if not os.path.exists(str(recog_cfg.get('index_path', '')).strip()):
                    reason.append('missing index_path')
                if not os.path.exists(str(recog_cfg.get('labels_path', '')).strip()):
                    reason.append('missing labels_path')
            except Exception:
                pass
            rs = (', '.join(reason) or 'not configured or unavailable')
            print(f"FAISS disabled: {rs}. Using Python matching.\n")
    except Exception as e:
        vector_index = None
        print(f"Vector index unavailable ({e}); fallback to Python matching.\n")

    save_feature = cfg['pipeline']['save_feature']
    save_path = None
    if save_feature:
        try:
            save_path = cfg['pipeline']['save_feature_path']
            if save_path:
                print(f"Features will save to {save_path}\n")
        except Exception as e:
            print("Unable to get save_feature_path\n")

    sources = cfg['source']
    number_sources = len(sources)

    Gst.init(None)
    print("Creating Pipeline \n ")
    pipeline = None
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    source_idx = 0
    is_live = False
    # for i in range(number_sources):
    for k, v in sources.items():
        print("Creating source_bin ", source_idx, " \n ")
        uri_name = v
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(source_idx, uri_name)

        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
            continue
        pipeline.add(source_bin)
        padname = "sink_%u" % source_idx
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
            continue
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
            continue
        srcpad.link(sinkpad)
        source_idx += 1

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")
    queue7 = Gst.ElementFactory.make("queue", "queue7")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    # Leaky queues to avoid backlog when syncing to clock; drop oldest buffers
    for q in (queue1, queue2, queue3, queue4, queue5, queue6, queue7):
        try:
            q.set_property('leaky', 2)  # downstream
            q.set_property('max-size-buffers', 10)
            q.set_property('max-size-bytes', 0)
            q.set_property('max-size-time', 0)
        except Exception:
            pass

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating Sgie \n ")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    if not sgie:
        sys.stderr.write(" Unable to create sgie \n")

    print("Creating Tracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Local sink only (remote streaming removed)
    sink = None
    # Detect display availability; fallback to fakesink on headless/TTY sessions
    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    headless = not has_display

    if not cfg['pipeline']['display'] or headless:
        if headless and cfg['pipeline']['display']:
            print("No GUI display detected (DISPLAY/WAYLAND_DISPLAY unset). Falling back to fakesink for headless run.\n")
        print("Creating Fakesink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        sink.set_property('enable-last-sample', 0)
        sink.set_property('sync', 0)
    else:
        if cfg['pipeline']['is_aarch64']:
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")

    if not sink:
        sys.stderr.write(" Unable to create sink element \n")

    if is_live:
        print("At least one of the sources is live \n")
        streammux.set_property('live-source', 1)

    set_property(cfg, streammux, "streammux")
    set_property(cfg, pgie, "pgie")
    set_property(cfg, sgie, "sgie")
    set_property(cfg, nvosd, "nvosd")
    # If headless, turn off text drawing to save cycles (can still attach meta if needed)
    try:
        if not cfg['pipeline']['display']:
            nvosd.set_property('display-text', 0)
    except Exception:
        pass
    set_property(cfg, tiler, "tiler")
    # Apply sink properties
    set_property(cfg, sink, "sink")
    # Decide realtime drop policy: if sink sync=1 or configured explicitly
    try:
        sink_sync = int(cfg.get('sink', {}).get('sync', 0)) == 1
    except Exception:
        sink_sync = False
    try:
        realtime_cfg = bool(cfg.get('pipeline', {}).get('realtime', 0))
    except Exception:
        realtime_cfg = False
    REALTIME_DROP = bool(sink_sync or realtime_cfg)
    if REALTIME_DROP and not is_live:
        print("Realtime mode: enabling streammux live-source and decoder frame dropping for file source\n")
        try:
            streammux.set_property('live-source', 1)
        except Exception:
            pass
    # If we sync to clock (even for file source), enable live-source so QoS can drop late frames
    try:
        if int(cfg.get('sink', {}).get('sync', 0)) == 1 and not is_live:
            print("Enabling streammux live-source for real-time pacing on file source\n")
            streammux.set_property('live-source', 1)
    except Exception:
        pass
    set_tracker_properties(tracker, cfg['tracker']['config-file-path'])

    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(tracker)
    if number_sources > 1:
        pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(sgie)
    sgie.link(queue4)
    if number_sources > 1:
        queue4.link(tiler)
        tiler.link(queue5)
    else:
        queue4.link(queue5)
    queue5.link(nvvidconv)
    nvvidconv.link(queue6)
    if cfg['pipeline']['display']:
        queue6.link(nvosd)
        nvosd.link(queue7)
        queue7.link(sink)
    else:
        # Bypass OSD in headless mode for better throughput
        queue6.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Optional timed stop for testing: quit loop after run_duration seconds
    if run_duration is not None:
        try:
            duration_ms = int(float(run_duration) * 1000)
        except Exception:
            duration_ms = None
        if duration_ms and duration_ms > 0:
            def _stop_loop():
                try:
                    loop.quit()
                except Exception:
                    pass
                return False  # do not repeat
            GLib.timeout_add(duration_ms, _stop_loop)

    print("Attach probes")
    pgie_src_pad = pgie.get_static_pad("src")
    # Note: pgie_src_filter_probe honors env vars to tune noise filtering without code change:
    #   PGIE_MIN_CONF (default 0.65), PGIE_MIN_W (20), PGIE_MIN_H (20),
    #   PGIE_MIN_AR (0.6), PGIE_MAX_AR (1.8), PGIE_TOPK (30)
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, 0)

    sgie_src_pad = sgie.get_static_pad("src")
    # Recognition settings
    try:
        recog_cfg = cfg.get('recognition', {})
        recog_thresh = recog_cfg.get('threshold', 0.3)
        recog_save_dir = recog_cfg.get('save_dir', '')
        recog_save_mode = str(recog_cfg.get('save_mode', 'all')).lower()
        recog_metric = str(recog_cfg.get('metric', 'cosine')).lower()
    except Exception:
        recog_thresh = 0.3
        recog_save_dir = ''
        recog_save_mode = 'all'
        recog_metric = 'cosine'
    # Fetch alignment pics dir from SGIE property (set via config)
    try:
        alignment_pic_dir = sgie.get_property('alignment-pic-path') or ''
    except Exception:
        alignment_pic_dir = ''
    # Debug/verbosity
    try:
        debug_cfg = cfg.get('debug', {})
        verbose = bool(debug_cfg.get('verbose', 0))
    except Exception:
        verbose = False

    data = [
        known_face_features,  # 0
        save_feature,         # 1
        save_path,            # 2 (feature npy dir)
        recog_thresh,         # 3
        alignment_pic_dir,    # 4
        recog_save_dir,       # 5 (recognized img dir)
        recog_save_mode,      # 6 (save mode: all|first|best)
        verbose,              # 7 (debug prints)
        vector_index,         # 8 (FAISS index or None)
        recog_metric          # 9 (metric: cosine|l2)
    ]
    sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, data)

    # List the sources
    print("Now playing...\n")
    for key, value in sources.items():
        print(f"{key}: {value} \n")

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    elapsed = time.time() - start_wall
    print(f"Wall elapsed: {elapsed:.2f} sec")
    try:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
    except Exception:
        pass


if __name__ == '__main__':
    cfg = parse_args(cfg_path="config/config_pipeline.toml")
    # Optional runtime cap to avoid indefinite hangs (env: DS_RUN_DURATION_SEC)
    run_dur = None
    try:
        _rd = os.getenv('DS_RUN_DURATION_SEC')
        if _rd:
            run_dur = float(_rd)
    except Exception:
        run_dur = None
    main(cfg, run_duration=run_dur)

