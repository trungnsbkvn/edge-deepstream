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
import math

from utils.probe import *
from utils.parser_cfg import *
from utils.bus_call import bus_call

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name)
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index,uri):
    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(cfg, run_duration=None):
    print(cfg)
    print("Load known faces features \n")
    known_face_features = load_faces(cfg['pipeline']['known_face_dir'])

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
    number_sources=len(sources)

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
        print("Creating source_bin ", source_idx," \n ")
        uri_name=v
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(source_idx, uri_name)
        
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" % source_idx
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
        source_idx += 1

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)

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
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
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

    # Remote streaming configuration
    remote_cfg = cfg.get('remote', {}) if isinstance(cfg, dict) else {}
    remote_enable = bool(remote_cfg.get('enable', 0))
    remote_host = remote_cfg.get('host', '127.0.0.1')
    remote_port = int(remote_cfg.get('port', 5600))
    remote_codec = str(remote_cfg.get('codec', 'h264')).lower()

    sink = None
    encoder = None
    parser = None
    mux = None
    udpsink = None
    postconv = None
    postcaps = None

    if remote_enable:
        print("Creating UDP streaming sink (MPEG-TS over UDP)\n")
        # Choose encoder per platform and codec
        if remote_codec == 'h265':
            enc_name_aarch64 = 'nvv4l2h265enc'
            enc_name_dgpu = 'nvh265enc'
            parser_name = 'h265parse'
        else:
            enc_name_aarch64 = 'nvv4l2h264enc'
            enc_name_dgpu = 'nvh264enc'
            parser_name = 'h264parse'

        enc_name = enc_name_aarch64 if cfg['pipeline']['is_aarch64'] else enc_name_dgpu
        encoder = Gst.ElementFactory.make(enc_name, 'encoder')
        # Fallback to software x264/x265 if HW encoder not available
        if not encoder and remote_codec == 'h264':
            encoder = Gst.ElementFactory.make('x264enc', 'encoder')
            if encoder:
                encoder.set_property('tune', 'zerolatency')
                encoder.set_property('speed-preset', 'superfast')
        if not encoder and remote_codec == 'h265':
            encoder = Gst.ElementFactory.make('x265enc', 'encoder')
            if encoder:
                encoder.set_property('tune', 'zerolatency')

        if not encoder:
            sys.stderr.write(" Unable to create encoder for remote streaming; falling back to fakesink\n")
        else:
            # Reasonable defaults
            if enc_name.startswith('nvv4l2'):
                encoder.set_property('bitrate', 4000000)  # ~4 Mbps
                if remote_codec == 'h264':
                    encoder.set_property('insert-sps-pps', 1)
                encoder.set_property('iframeinterval', 30)
                # 1 = VBR, 2 = CBR
                encoder.set_property('control-rate', 1)
            elif enc_name.startswith('nvh'):
                # nvh264enc/nvh265enc properties
                if encoder.find_property('bitrate'):
                    encoder.set_property('bitrate', 4000)  # in kbps
                if encoder.find_property('preset'):
                    encoder.set_property('preset', 'low-latency-hq')

            parser = Gst.ElementFactory.make(parser_name, 'parser')
            mux = Gst.ElementFactory.make('mpegtsmux', 'tsmux')
            udpsink = Gst.ElementFactory.make('udpsink', 'udpsink')
            if udpsink:
                udpsink.set_property('host', remote_host)
                udpsink.set_property('port', remote_port)
                udpsink.set_property('sync', 0)
                udpsink.set_property('async', 0)

            # Add post-OSD converter and caps to feed encoder NV12
            postconv = Gst.ElementFactory.make('nvvideoconvert', 'postconvert')
            postcaps = Gst.ElementFactory.make('capsfilter', 'postcaps')
            if postcaps:
                from gi.repository import Gst as _Gst
                caps_str = 'video/x-raw(memory:NVMM), format=NV12' if cfg['pipeline']['is_aarch64'] else 'video/x-raw, format=NV12'
                postcaps.set_property('caps', _Gst.Caps.from_string(caps_str))

            if not (parser and mux and udpsink and postconv and postcaps):
                sys.stderr.write(" Unable to create streaming elements; falling back to fakesink\n")
                encoder = None

    if not remote_enable or encoder is None:
        # Use local display or fakesink
        if not cfg['pipeline']['display']:
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
    set_property(cfg, tiler, "tiler")
    # Apply sink properties only for local sink path
    if not (remote_enable and encoder is not None):
        set_property(cfg, sink, "sink")
    set_tracker_properties(tracker, cfg['tracker']['config-file-path'])

    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    if remote_enable and encoder is not None:
        pipeline.add(postconv)
        pipeline.add(postcaps)
        pipeline.add(encoder)
        pipeline.add(parser)
        pipeline.add(mux)
        pipeline.add(udpsink)
    else:
        pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(sgie)
    sgie.link(queue4)
    queue4.link(tiler)
    tiler.link(queue5)
    queue5.link(nvvidconv)
    nvvidconv.link(queue6)
    queue6.link(nvosd)
    nvosd.link(queue7)
    if remote_enable and encoder is not None:
        # nvosd -> queue7 -> nvvideoconvert -> caps -> encoder -> parser -> mpegtsmux -> udpsink
        if not queue7.link(postconv):
            sys.stderr.write(" Failed to link postconvert \n")
        if not postconv.link(postcaps):
            sys.stderr.write(" Failed to link postcaps \n")
        if not postcaps.link(encoder):
            sys.stderr.write(" Failed to link encoder \n")
        if not encoder.link(parser):
            sys.stderr.write(" Failed to link parser \n")
        if not parser.link(mux):
            sys.stderr.write(" Failed to link mpegtsmux \n")
        if not mux.link(udpsink):
            sys.stderr.write(" Failed to link udpsink \n")
    else:
        queue7.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

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
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, 0)

    sgie_src_pad = sgie.get_static_pad("src")
    # Recognition settings
    try:
        recog_thresh = cfg.get('recognition', {}).get('threshold', 0.3)
    except Exception:
        recog_thresh = 0.3
    data = [known_face_features, save_feature, save_path, recog_thresh]
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
    try:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
    except Exception:
        pass

if __name__ == '__main__':
    cfg = parse_args(cfg_path="config/config_pipeline.toml")
    main(cfg)

