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
import json

from utils.probe import *
from utils.parser_cfg import *
from utils.bus_call import bus_call
from utils.event_sender import EventSender
from utils.mqtt_listener import MQTTListener
from utils.faiss_index import FaceIndex, FaceIndexConfig
from utils.gen_feature import TensorRTInfer
from typing import Any, Dict, Optional
import configparser

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
        # Low-latency tuning for RTSP source (rtspsrc)
        try:
            env_latency = os.getenv('DS_RTSP_LATENCY')
            latency_val = int(env_latency) if env_latency and env_latency.isdigit() else 150
            if Object.find_property('latency') is not None:
                Object.set_property('latency', latency_val)
            if os.getenv('DS_RTSP_TCP','0') == '1' and Object.find_property('protocols') is not None:
                try:
                    Object.set_property('protocols', 4)  # TCP
                except Exception:
                    pass
            if Object.find_property('do-retransmission') is not None:
                Object.set_property('do-retransmission', os.getenv('DS_RTSP_RETRANS','0') == '1')
            if Object.find_property('drop-on-latency') is not None:
                Object.set_property('drop-on-latency', os.getenv('DS_RTSP_DROP_ON_LATENCY','0') == '1')
            if Object.find_property('ntp-sync') is not None:
                Object.set_property('ntp-sync', False)
        except Exception:
            pass
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
    # Explicit RTSP (H264) path to avoid uridecodebin negotiation issues
    if uri.startswith('rtsp://'):
        try:
            print(f"[RTSP] (explicit) Creating RTSP source index={index} uri={uri}")
            rtspsrc = Gst.ElementFactory.make('rtspsrc', f'rtspsrc-{index}')
            if not rtspsrc:
                raise RuntimeError('rtspsrc create failed')
            rtspsrc.set_property('location', uri)
            # Latency/env overrides
            try:
                env_latency = os.getenv('DS_RTSP_LATENCY')
                latency_val = int(env_latency) if env_latency and env_latency.isdigit() else 150
                if rtspsrc.find_property('latency') is not None:
                    rtspsrc.set_property('latency', latency_val)
            except Exception:
                pass
            if os.getenv('DS_RTSP_TCP','0') == '1' and rtspsrc.find_property('protocols') is not None:
                try:
                    rtspsrc.set_property('protocols', 4)  # TCP
                except Exception:
                    pass
            depay = Gst.ElementFactory.make('rtph264depay', f'depay-{index}')
            h264parse = Gst.ElementFactory.make('h264parse', f'h264parse-{index}')
            decoder = None
            # Prefer hardware decoder
            for cand in ['nvv4l2decoder', 'nvh264dec', 'avdec_h264']:
                decoder = Gst.ElementFactory.make(cand, f'dec-{cand}-{index}')
                if decoder:
                    break
            if not (depay and h264parse and decoder):
                raise RuntimeError('H264 decode chain creation failed')
            queue_post = Gst.ElementFactory.make('queue', f'queue-postdec-{index}')
            if queue_post:
                try:
                    queue_post.set_property('leaky', 2)
                    queue_post.set_property('max-size-buffers', 10)
                except Exception:
                    pass
            for e in [rtspsrc, depay, h264parse, decoder, queue_post]:
                if e:
                    nbin.add(e)

            def _rtsp_pad_added(src, pad):
                try:
                    caps = pad.get_current_caps()
                    name = caps.to_string() if caps else ''
                    # Link only RTP H264 pads
                    if 'application/x-rtp' in name and 'H264' in name.upper():
                        sinkpad = depay.get_static_pad('sink')
                        if sinkpad and pad.can_link(sinkpad):
                            pad.link(sinkpad)
                except Exception:
                    pass

            rtspsrc.connect('pad-added', _rtsp_pad_added)
            # Link static parts
            if not depay.link(h264parse):
                print(f"[RTSP] depay->parse link failed index={index}")
            if not h264parse.link(decoder):
                print(f"[RTSP] parse->decoder link failed index={index}")
            if queue_post and not decoder.link(queue_post):
                print(f"[RTSP] decoder->queue link failed index={index}")
            # Create ghost pad from last element src
            last = queue_post or decoder
            ghost_src = last.get_static_pad('src')
            if not ghost_src:
                raise RuntimeError('No src pad for ghost')
            if not nbin.add_pad(Gst.GhostPad.new('src', ghost_src)):
                raise RuntimeError('Ghost pad add failed')
            return nbin
        except Exception as e:
            try:
                print(f"[RTSP] explicit path failed ({e}); falling back to uridecodebin")
            except Exception:
                pass
            # Fall through to uridecodebin

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    uri_decode_bin.set_property("uri", uri)
    try:
        force_nvmm = os.getenv('DS_FORCE_NVMM', '0') == '1'
        if force_nvmm:
            print(f"[RTSP] Forcing NVMM caps for source index={index}")
            uri_decode_bin.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        if uri_decode_bin.find_property('expose-all-streams') is not None:
            uri_decode_bin.set_property('expose-all-streams', False)
    except Exception as e:
        try:
            print(f"[RTSP] caps setup warning: {e}")
        except Exception:
            pass
    try:
        print(f"[RTSP] Creating source index={index} uri={uri}")
    except Exception:
        pass
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
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
    # Build a deterministic mapping from batch index to cameraId using [source] keys directly
    source_index_to_cam = {}
    # Registry for dynamic management: cam_id -> { 'bin', 'sinkpad', 'index', 'uri' }
    source_registry = {}

    # Track dynamic batch mode (batch-size <=0 means auto-grow, keep fixed max)
    dynamic_batch_mode = False
    dynamic_batch_target_max = 0

    def _sanitize_uri(u: str) -> str:
        try:
            if not u:
                return u
            s = u.strip().strip('"').strip("'")
            s = ''.join(ch for ch in s if ord(ch) >= 32)
            if ' ' in s:
                s = s.replace(' ', '')
            return s
        except Exception:
            return u

    # Helper to (re)compute tiler layout when number of sources changes
    def _update_tiler_layout(total_sources: int):
        try:
            if total_sources < 1:
                total_sources = 1
            rows = int(math.sqrt(total_sources))
            cols = int(math.ceil((1.0 * total_sources) / rows))
            try:
                tiler.set_property("rows", rows)
                tiler.set_property("columns", cols)
            except Exception:
                pass
        except Exception:
            pass

    # Compute batch-size as (max index + 1) to preserve source_id mapping when indices have holes
    def _active_batch_size() -> int:
        try:
            if not source_registry:
                return 0
            max_idx = max(int(e.get('index', -1)) for e in source_registry.values())
            return max_idx + 1
        except Exception:
            return len(source_registry)

    # Comment out a source entry in the TOML [source] table (best-effort)
    def _comment_source_in_config(cam_id: str):
        try:
            cfg_path = os.getenv('DS_CONFIG_PATH', 'config/config_pipeline.toml')
            if not os.path.exists(cfg_path):
                return
            with open(cfg_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            in_source = False
            key_prefix = f'"{cam_id}"'
            changed = False
            new_lines = []
            for ln in lines:
                s = ln.strip()
                if s.startswith('[source]'):
                    in_source = True
                    new_lines.append(ln)
                    continue
                if in_source and s.startswith('[') and s.endswith(']'):
                    in_source = False
                    new_lines.append(ln)
                    continue
                if in_source and s.startswith(key_prefix) and not s.startswith('#'):
                    new_lines.append('# ' + ln)
                    changed = True
                    continue
                new_lines.append(ln)
            if changed:
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
        except Exception:
            pass

    # Persist (add or update) a source entry in the TOML [source] table (best-effort)
    def _persist_source_update(cam_id: str, uri: str):
        # Re-write [source] section fully (sorted, compact) and update [streammux].batch-size
        try:
            cfg_path = os.getenv('DS_CONFIG_PATH', 'config/config_pipeline.toml')
            if not os.path.exists(cfg_path):
                return
            try:
                # Use current runtime registry for authoritative list (active sources only)
                active_sources = {cid: entry.get('uri','') for cid, entry in source_registry.items()}
                # Ensure the just-updated/new cam_id is present (in case called early)
                if cam_id and uri:
                    active_sources[str(cam_id)] = uri
            except Exception:
                active_sources = {cam_id: uri} if cam_id and uri else {}

            with open(cfg_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()

            new_lines = []
            in_source = False
            in_streammux = False
            batch_written = False
            # Determine batch-size to persist: number of active sources (so next start uses deterministic size)
            try:
                persist_batch = len([k for k,v in active_sources.items() if v])
            except Exception:
                persist_batch = 0

            for idx, ln in enumerate(original_lines):
                stripped = ln.strip()
                # Section starts
                if stripped.startswith('[') and stripped.endswith(']'):
                    # Close any open section specific state
                    if in_streammux and not batch_written:
                        new_lines.append(f"batch-size={persist_batch}\n")
                    in_streammux = False
                    in_source = False
                    # Begin new section
                    if stripped == '[source]':
                        in_source = True
                        new_lines.append('[source]\n')
                        # Immediately write sorted active sources (skip commented / old ones)
                        for cid in sorted(active_sources.keys()):
                            val = active_sources[cid]
                            if val:
                                new_lines.append(f'"{cid}" = "{val}"\n')
                        continue  # Skip writing original source section contents
                    elif stripped == '[streammux]':
                        in_streammux = True
                        batch_written = False
                        new_lines.append(ln)
                        continue
                    else:
                        new_lines.append(ln)
                        continue

                # Inside [source] original section: skip its lines (already rewritten)
                if in_source:
                    continue

                if in_streammux:
                    # Replace any existing batch-size line
                    if stripped.startswith('batch-size'):
                        if not batch_written:
                            new_lines.append(f"batch-size={persist_batch}\n")
                            batch_written = True
                        continue
                    # Normal line inside streammux
                    new_lines.append(ln)
                    continue

                # Default passthrough
                new_lines.append(ln)

            # File ended while inside streammux section without batch-size
            if in_streammux and not batch_written:
                new_lines.append(f"batch-size={persist_batch}\n")

            # Ensure a [source] section exists if it was missing originally
            if '[source]' not in ''.join(new_lines):
                new_lines.append('\n[source]\n')
                for cid in sorted(active_sources.keys()):
                    val = active_sources[cid]
                    if val:
                        new_lines.append(f'"{cid}" = "{val}"\n')

            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception:
            pass

    # Thread-safe add/update of a source while pipeline is running
    def _add_or_update_source(cam_id: str, uri: str, resp_meta: Optional[dict] = None):
        nonlocal source_idx, is_live
        try:
            cid = str(cam_id)
            if not cid:
                return False
            uri = _sanitize_uri(uri)
            if uri and uri.startswith('rtsp://'):
                is_live = True
                try:
                    streammux.set_property('live-source', 1)
                except Exception:
                    pass
            # Update existing
            if cid in source_registry:
                entry = source_registry[cid]
                if entry.get('uri') == uri:
                    if resp_meta:
                        try:
                            _publish_response(resp_meta.get('cmd',0), cid, resp_meta.get('user_id',''), resp_meta.get('cmd_id',''), resp_meta.get('status',0))
                        except Exception:
                            pass
                    return False
                print(f"[MQTT] Updating source for cam {cid}")
                # Tear down old bin
                try:
                    old_bin = entry.get('bin')
                    old_pad = entry.get('sinkpad')
                    if old_pad:
                        try:
                            old_pad.send_event(Gst.Event.new_eos())
                        except Exception:
                            pass
                        try:
                            streammux.release_request_pad(old_pad)
                        except Exception:
                            pass
                    if old_bin:
                        try:
                            old_bin.set_state(Gst.State.NULL)
                        except Exception:
                            pass
                        try:
                            pipeline.remove(old_bin)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Recreate with same index
                idx = int(entry.get('index', source_idx))
                new_bin = create_source_bin(idx, uri)
                if not new_bin:
                    print(f"[MQTT] Failed to create source bin for {cid}")
                    return False
                pipeline.add(new_bin)
                padname = f"sink_{idx}"
                sinkpad = streammux.get_request_pad(padname)
                if not sinkpad:
                    print(f"[MQTT] Failed to get streammux pad {padname}")
                    try:
                        pipeline.remove(new_bin)
                    except Exception:
                        pass
                    return False
                srcpad = new_bin.get_static_pad("src")
                if not srcpad:
                    print("[MQTT] Failed to get src pad from new source bin")
                    try:
                        streammux.release_request_pad(sinkpad)
                    except Exception:
                        pass
                    try:
                        pipeline.remove(new_bin)
                    except Exception:
                        pass
                    return False
                srcpad.link(sinkpad)
                try:
                    new_bin.sync_state_with_parent()
                except Exception:
                    try:
                        new_bin.set_state(Gst.State.PLAYING)
                    except Exception:
                        pass
                # Update registry and mapping
                entry.update({'bin': new_bin, 'sinkpad': sinkpad, 'uri': uri})
                entry['last_frame_ts'] = time.time()
                entry['reconnect_attempts'] = 0
                try:
                    if dynamic_batch_mode:
                        streammux.set_property('batch-size', min(dynamic_batch_target_max, max(1, len(source_registry))))
                    else:
                        streammux.set_property('batch-size', _active_batch_size())
                except Exception:
                    pass
                _update_tiler_layout(len(source_registry))
                if resp_meta:
                    try:
                        _publish_response(resp_meta.get('cmd',0), cid, resp_meta.get('user_id',''), resp_meta.get('cmd_id',''), resp_meta.get('status',0))
                    except Exception:
                        pass
                try:
                    _persist_source_update(cid, uri)
                except Exception:
                    pass
                return False

            # Add new
            print(f"[MQTT] Adding new source cam {cid}")
            idx = int(source_idx)
            new_bin = create_source_bin(idx, uri)
            if not new_bin:
                print(f"[MQTT] Failed to create source bin for {cid}")
                return False
            pipeline.add(new_bin)
            padname = f"sink_{idx}"
            sinkpad = streammux.get_request_pad(padname)
            if not sinkpad:
                print(f"[MQTT] Failed to get streammux pad {padname}")
                try:
                    pipeline.remove(new_bin)
                except Exception:
                    pass
                return False
            srcpad = new_bin.get_static_pad("src")
            if not srcpad:
                print("[MQTT] Failed to get src pad from source bin")
                try:
                    streammux.release_request_pad(sinkpad)
                except Exception:
                    pass
                try:
                    pipeline.remove(new_bin)
                except Exception:
                    pass
                return False
            srcpad.link(sinkpad)
            try:
                new_bin.sync_state_with_parent()
            except Exception:
                try:
                    new_bin.set_state(Gst.State.PLAYING)
                except Exception:
                    pass
            # Update maps
            source_index_to_cam[idx] = cid
            source_registry[cid] = {'bin': new_bin, 'sinkpad': sinkpad, 'index': idx, 'uri': uri}
            try:
                source_registry[cid]['last_frame_ts'] = time.time()
                source_registry[cid]['reconnect_attempts'] = 0
            except Exception:
                pass
            source_idx += 1
            try:
                if dynamic_batch_mode:
                    streammux.set_property('batch-size', min(dynamic_batch_target_max, max(1, len(source_registry))))
                else:
                    streammux.set_property('batch-size', _active_batch_size())
            except Exception:
                pass
            _update_tiler_layout(len(source_registry))
            try:
                print(f"[MQTT] Source added: cam_id={cid} index={idx} total_active={len(source_registry)} batch_size={_active_batch_size()}")
            except Exception:
                pass
            if resp_meta:
                try:
                    _publish_response(resp_meta.get('cmd',0), cid, resp_meta.get('user_id',''), resp_meta.get('cmd_id',''), resp_meta.get('status',0))
                except Exception:
                    pass
            try:
                _persist_source_update(cid, uri)
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"[MQTT] add/update error: {e}")
            except Exception:
                pass
        return False

    def _remove_source(cam_id: str, resp_meta: Optional[dict] = None):
        try:
            cid = str(cam_id)
            if cid not in source_registry:
                return False
            print(f"[MQTT] Removing source cam {cid}")
            entry = source_registry.pop(cid, None)
            if not entry:
                return False
            # Release streammux pad and remove bin
            try:
                pad = entry.get('sinkpad')
                if pad:
                    try:
                        pad.send_event(Gst.Event.new_eos())
                    except Exception:
                        pass
                    try:
                        streammux.release_request_pad(pad)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                bin_ = entry.get('bin')
                if bin_:
                    try:
                        bin_.set_state(Gst.State.NULL)
                    except Exception:
                        pass
                    try:
                        pipeline.remove(bin_)
                    except Exception:
                        pass
            except Exception:
                pass
            # Clear index mapping for removed slot
            try:
                idx = int(entry.get('index', -1))
                if idx in source_index_to_cam:
                    source_index_to_cam.pop(idx, None)
            except Exception:
                pass
            # Update batch-size to remaining sources (preserve mapping)
            try:
                if dynamic_batch_mode:
                    streammux.set_property('batch-size', min(dynamic_batch_target_max, max(1, len(source_registry))))
                else:
                    streammux.set_property('batch-size', _active_batch_size())
            except Exception:
                pass
            _update_tiler_layout(len(source_registry))
            # Comment the source in config for persistence
            _comment_source_in_config(cid)
            try:
                _persist_source_update('', '')  # Re-write config to update batch-size
            except Exception:
                pass
            if resp_meta:
                try:
                    _publish_response(resp_meta.get('cmd',0), cid, resp_meta.get('user_id',''), resp_meta.get('cmd_id',''), resp_meta.get('status',0))
                except Exception:
                    pass
        except Exception as e:
            try:
                print(f"[MQTT] remove error: {e}")
            except Exception:
                pass
        return False
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
        # Record mapping (k is camera id string from config)
        try:
            source_index_to_cam[source_idx] = str(k)
        except Exception:
            source_index_to_cam[source_idx] = str(source_idx)
        # Track registry for dynamic mgmt
        source_registry[str(k)] = {
            'bin': source_bin,
            'sinkpad': sinkpad,
            'index': int(source_idx),
            'uri': uri_name
        }
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
    # Leaky queues to avoid backlog; drop oldest buffers to prevent visual repeats
    for q in (queue1, queue2, queue3, queue4, queue5, queue6, queue7):
        try:
            q.set_property('leaky', 2)  # downstream: drop oldest when downstream is slow
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
    # After applying configured properties, evaluate dynamic batch-size mode
    try:
        cfg_bs = int(cfg.get('streammux', {}).get('batch-size', 0))
    except Exception:
        cfg_bs = 0
    if cfg_bs <= 0:
        dynamic_batch_mode = True
        try:
            dynamic_batch_target_max = int(os.getenv('DS_DYNAMIC_MAX_BATCH', '16'))
        except Exception:
            dynamic_batch_target_max = 16
        if dynamic_batch_target_max < 1:
            dynamic_batch_target_max = 1
        try:
            streammux.set_property('batch-size', 1)
            print(f"[DYN] streammux dynamic mode: config batch-size={cfg_bs}; target_max={dynamic_batch_target_max} start=1")
        except Exception:
            pass
    else:
        dynamic_batch_mode = False
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

    # Handle zero-initial sources safely
    try:
        init_sources = max(1, number_sources)
        tiler_rows = int(math.sqrt(init_sources))
        tiler_columns = int(math.ceil((1.0 * init_sources) / tiler_rows))
    except Exception:
        tiler_rows, tiler_columns = 1, 1
    try:
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
    except Exception:
        pass
    # If starting with zero sources, ensure streammux batch-size at least 1; will grow dynamically
    try:
        if number_sources == 0:
            streammux.set_property('batch-size', 1)
    except Exception:
        pass

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(tracker)
    # Always add tiler so dynamic source additions >1 are supported
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
    queue4.link(tiler)
    tiler.link(queue5)
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
    # Build probe data with cfg for thresholds; env can still override in probe
    pgie_probe_data = {
        'thresholds': cfg.get('thresholds', {})
    }
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, pgie_probe_data)

    sgie_src_pad = sgie.get_static_pad("src")
    # Recognition settings
    try:
        recog_cfg = cfg.get('recognition', {})
        recog_thresh = recog_cfg.get('threshold', 0.3)
        # Env override for quick debug without editing config
        _rt_env = os.getenv('RECOG_THRESH')
        if _rt_env is not None:
            try:
                recog_thresh = float(_rt_env)
            except Exception:
                pass
        recog_save_dir = recog_cfg.get('save_dir', '')
        recog_save_mode = str(recog_cfg.get('save_mode', 'all')).lower()
        recog_metric = str(recog_cfg.get('metric', 'cosine')).lower()
    except Exception:
        recog_thresh = 0.3
        recog_save_dir = ''
        recog_save_mode = 'all'
        recog_metric = 'cosine'
    # Fetch alignment pics dir from SGIE property (set via config)
    # Note: custom property name is 'alignment-pics-path' (plural). Keep a fallback to singular.
    try:
        alignment_pic_dir = ''
        try:
            alignment_pic_dir = sgie.get_property('alignment-pics-path') or ''
        except Exception:
            alignment_pic_dir = ''
        if not alignment_pic_dir:
            # Fallback for older builds using singular form
            try:
                alignment_pic_dir = sgie.get_property('alignment-pic-path') or ''
            except Exception:
                alignment_pic_dir = ''
    except Exception:
        alignment_pic_dir = ''
    # Debug/verbosity
    try:
        debug_cfg = cfg.get('debug', {})
        verbose = bool(debug_cfg.get('verbose', 0))
    except Exception:
        verbose = False

    # Indexing options for live updates
    try:
        idx_enable = int(recog_cfg.get('index_stream', 0)) == 1
    except Exception:
        idx_enable = False
    try:
        idx_mode = str(recog_cfg.get('index_mode', 'per_track')).lower()
    except Exception:
        idx_mode = 'per_track'
    try:
        idx_label = str(recog_cfg.get('index_label', 'track')).lower()
    except Exception:
        idx_label = 'track'
    try:
        index_path_cfg = str(recog_cfg.get('index_path', '')).strip()
        labels_path_cfg = str(recog_cfg.get('labels_path', '')).strip()
        stream_index_path_cfg = str(recog_cfg.get('stream_index_path', '')).strip()
        stream_labels_path_cfg = str(recog_cfg.get('stream_labels_path', '')).strip()
        if idx_enable:
            idx_path = stream_index_path_cfg or index_path_cfg
            lbl_path = stream_labels_path_cfg or labels_path_cfg
        else:
            # When live indexing is disabled, ensure we read the main labels.json for name mapping
            idx_path = index_path_cfg
            lbl_path = labels_path_cfg
    except Exception:
        idx_path, lbl_path = '', ''
    try:
        recognize_once = int(recog_cfg.get('recognize_once_per_track', 1)) == 1
    except Exception:
        recognize_once = True

    # Event sink via Unix domain socket (optional)
    try:
        evt_cfg = cfg.get('events', {})
        socket_path = str(evt_cfg.get('unix_socket', '/tmp/my_socket')).strip()
        send_images = int(evt_cfg.get('send_image', 1)) == 1
        enabled_evt = int(evt_cfg.get('enable', 1)) == 1
    except Exception:
        socket_path, send_images, enabled_evt = '/tmp/my_socket', True, True
    event_sender = EventSender(socket_path) if enabled_evt and socket_path else None

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
        recog_metric,         # 9 (metric: cosine|l2)
        idx_enable,           # 10 live indexing enabled
        idx_mode,             # 11 indexing mode: per_track|per_frame
        idx_label,            # 12 index label mode: track|name
        idx_path,             # 13 index save path
        lbl_path,             # 14 labels save path
        recognize_once,       # 15 recognize once per track
        source_index_to_cam,  # 16 map: mux batch index -> cameraId
        event_sender,         # 17 EventSender or None
        send_images           # 18 send image bytes flag
    ]
    sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, data)

    # --- Frame activity probe for RTSP health monitoring (attach early in pipeline) ---
    try:
        # Probe on queue1 (after streammux) to inspect batch meta for all frames
        q1_src_pad = queue1.get_static_pad('src')
        if q1_src_pad:
            def _activity_probe(pad, info, udata):
                try:
                    buf = info.get_buffer()
                    if not buf:
                        return Gst.PadProbeReturn.OK
                    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
                    if not batch_meta:
                        return Gst.PadProbeReturn.OK
                    l_frame = batch_meta.frame_meta_list
                    while l_frame is not None:
                        try:
                            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                        except Exception:
                            frame_meta = None
                        if frame_meta:
                            sid = int(frame_meta.source_id)
                            cam_id = source_index_to_cam.get(sid)
                            if cam_id and cam_id in source_registry:
                                e = source_registry[cam_id]
                                e['last_frame_ts'] = time.time()
                                e['reconnect_attempts'] = 0
                        try:
                            l_frame = l_frame.next
                        except Exception:
                            break
                except Exception:
                    pass
                return Gst.PadProbeReturn.OK
            q1_src_pad.add_probe(Gst.PadProbeType.BUFFER, _activity_probe, None)
    except Exception:
        pass

    # --- RTSP health watchdog ---
    def _force_reconnect_source(cam_id: str):
        try:
            if cam_id not in source_registry:
                return False
            entry = source_registry[cam_id]
            uri = entry.get('uri')
            idx = int(entry.get('index', -1))
            if idx < 0 or not uri:
                return False
            print(f"[RTSP] Reconnecting cam {cam_id} (index {idx}) uri={uri}")
            # Tear down existing
            try:
                old_pad = entry.get('sinkpad')
                if old_pad:
                    # Don't push EOS (single-source pipeline would shut down)
                    try:
                        streammux.release_request_pad(old_pad)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                old_bin = entry.get('bin')
                if old_bin:
                    try:
                        old_bin.set_state(Gst.State.NULL)
                    except Exception:
                        pass
                    try:
                        pipeline.remove(old_bin)
                    except Exception:
                        pass
            except Exception:
                pass
            # Create new bin
            new_bin = create_source_bin(idx, uri)
            if not new_bin:
                print(f"[RTSP] Failed to create new bin for cam {cam_id}")
                return False
            pipeline.add(new_bin)
            padname = f"sink_{idx}"
            sinkpad = streammux.get_request_pad(padname)
            if not sinkpad:
                print(f"[RTSP] Failed to get sink pad {padname} for cam {cam_id}")
                try:
                    pipeline.remove(new_bin)
                except Exception:
                    pass
                return False
            srcpad = new_bin.get_static_pad("src")
            if not srcpad:
                print(f"[RTSP] Missing src pad for cam {cam_id}")
                try:
                    streammux.release_request_pad(sinkpad)
                except Exception:
                    pass
                try:
                    pipeline.remove(new_bin)
                except Exception:
                    pass
                return False
            srcpad.link(sinkpad)
            try:
                new_bin.sync_state_with_parent()
            except Exception:
                try:
                    new_bin.set_state(Gst.State.PLAYING)
                except Exception:
                    pass
            entry.update({'bin': new_bin, 'sinkpad': sinkpad})
            entry['last_frame_ts'] = time.time()
            return False
        except Exception as e:
            try:
                print(f"[RTSP] reconnect error: {e}")
            except Exception:
                pass
        return False

    try:
        interval_ms = int(os.getenv('DS_HEALTH_INTERVAL_MS', '5000'))
    except Exception:
        interval_ms = 5000
    try:
        timeout_sec = float(os.getenv('DS_HEALTH_TIMEOUT_SEC', '20'))
    except Exception:
        timeout_sec = 20.0
    try:
        max_retries = int(os.getenv('DS_HEALTH_MAX_RETRIES', '3'))
    except Exception:
        max_retries = 3

    def _watchdog():
        now = time.time()
        for cid, entry in list(source_registry.items()):
            try:
                uri = entry.get('uri','')
                if not uri.startswith('rtsp://'):
                    continue
                last_ts = entry.get('last_frame_ts', 0)
                if last_ts == 0:
                    # Initial grace period (2 * timeout) before first reconnect
                    entry['last_frame_ts'] = now
                    if now - start_wall < timeout_sec * 2:
                        continue
                if last_ts and (now - last_ts) > (timeout_sec / 2.0) and entry.get('reconnect_attempts',0) == 0:
                    # Half-timeout info log to show negotiation delay
                    if entry.get('logged_half', False) is False:
                        print(f"[RTSP] cam {cid} no frames yet ({now-last_ts:.1f}s) awaiting first frame")
                        entry['logged_half'] = True
                if now - last_ts > timeout_sec:
                    attempts = int(entry.get('reconnect_attempts', 0))
                    if attempts >= max_retries:
                        if attempts == max_retries:
                            print(f"[RTSP] cam {cid} exceeded max retries ({max_retries}); giving up.")
                            entry['reconnect_attempts'] = attempts + 1  # avoid repeating log
                        continue
                    print(f"[RTSP] cam {cid} stale ({now-last_ts:.1f}s > {timeout_sec}s); reconnect attempt {attempts+1}")
                    entry['reconnect_attempts'] = attempts + 1
                    GLib.idle_add(_force_reconnect_source, cid)
            except Exception:
                pass
        return True  # repeat

    if interval_ms > 0 and timeout_sec > 0:
        try:
            GLib.timeout_add(interval_ms, _watchdog)
            print(f"[RTSP] Health watchdog active: interval={interval_ms}ms timeout={timeout_sec}s max_retries={max_retries}")
        except Exception:
            pass

    # List the sources
    if number_sources == 0:
        print("No initial sources configured. Waiting for MQTT enable-service (cmd 25) messages to add sources...\n")
    print("Now playing...\n")
    for key, value in sources.items():
        print(f"{key}: {value} \n")

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)

    # --- Helper: load SGIE engine for enrollment (lazy) ---
    _trt_model: Dict[str, Any] = {'inst': None}
    def _get_trt_model():
        if _trt_model['inst'] is not None:
            return _trt_model['inst']
        try:
            # Resolve engine from SGIE config
            sgie_cfg_path = cfg.get('sgie', {}).get('config-file-path', '').strip()
            eng = None
            if sgie_cfg_path and os.path.exists(sgie_cfg_path):
                cp = configparser.ConfigParser()
                cp.read(sgie_cfg_path)
                if cp.has_option('property', 'model-engine-file'):
                    eng = cp.get('property', 'model-engine-file')
                    if eng and not os.path.isabs(eng):
                        eng = os.path.normpath(os.path.join(os.path.dirname(sgie_cfg_path), eng))
            if not eng or not os.path.exists(eng):
                # Fallbacks
                cand = os.path.join('models', 'arcface', 'arcface.engine')
                eng = cand
            inst = TensorRTInfer(eng, mode='min')
            _trt_model['inst'] = inst
            return inst
        except Exception as e:
            print(f"[ENROLL] TRT engine load failed: {e}")
            return None

    # --- Helper: simple face alignment to ArcFace 112x112 ---
    def _align_arcface_from_image_path(img_path: str):
        import cv2
        bgr = cv2.imread(img_path)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        # If already 112x112, just convert to RGB
        if h == 112 and w == 112:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Detect face using Haar cascade (fallback: center crop)
        try:
            cascade_dir = getattr(cv2, 'data', None)
            cascade_path = None
            if cascade_dir is not None and hasattr(cascade_dir, 'haarcascades'):
                cascade_path = os.path.join(cascade_dir.haarcascades, 'haarcascade_frontalface_default.xml')
            else:
                # Fallback common locations
                for p in [
                    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                ]:
                    if os.path.exists(p):
                        cascade_path = p
                        break
            face_cascade = cv2.CascadeClassifier(cascade_path) if cascade_path else cv2.CascadeClassifier()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if faces is not None and len(faces) > 0:
                # Pick largest face
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                # Expand and make square
                cx = x + fw / 2.0
                cy = y + fh / 2.0
                side = int(max(fw, fh) * 1.25)
                sx = int(max(0, cx - side / 2))
                sy = int(max(0, cy - side / 2))
                ex = int(min(w, sx + side))
                ey = int(min(h, sy + side))
                crop = bgr[sy:ey, sx:ex]
            else:
                # Center square crop
                side = min(h, w)
                sx = (w - side) // 2
                sy = (h - side) // 2
                crop = bgr[sy:sy+side, sx:sx+side]
        except Exception:
            side = min(h, w)
            sx = (w - side) // 2
            sy = (h - side) // 2
            crop = bgr[sy:sy+side, sx:sx+side]
        # Resize to 112x112 and convert to RGB
        try:
            resized = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = cv2.resize(crop, (112, 112))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # --- Helper: ensure a mutable FaceIndex (build from existing or new) ---
    def _ensure_index() -> FaceIndex:
        nonlocal vector_index
        # Update the probe data reference (index 8) after creation/load so recognition uses latest
        try:
            if vector_index is not None:
                try:
                    if isinstance(data, list) and len(data) > 8:
                        data[8] = vector_index
                except Exception:
                    pass
                return vector_index
        except Exception:
            pass
        # Try to load from disk
        vi = safe_load_index(recog_cfg)
        if vi is not None:
            vector_index = vi
            try:
                if isinstance(data, list) and len(data) > 8:
                    data[8] = vector_index
            except Exception:
                pass
            return vector_index
        # Else, create empty index with default dim 512 for ArcFace
        try:
            cfg_idx = FaceIndexConfig(metric=recog_metric, index_type=recog_cfg.get('index_type', 'flat'), use_gpu=bool(int(recog_cfg.get('use_gpu', 0))), gpu_id=int(recog_cfg.get('gpu_id', 0)))
        except Exception:
            cfg_idx = FaceIndexConfig(metric='cosine', index_type='flat', use_gpu=False, gpu_id=0)
        idx = FaceIndex(dim=512, cfg=cfg_idx)
        # Build empty CPU/GPU handles
        cpu_index, _ = idx._make_faiss_index(0)
        idx._cpu_index = cpu_index
        idx._gpu_index = idx._to_gpu(cpu_index)
        idx._labels = []
        vector_index = idx
        try:
            if isinstance(data, list) and len(data) > 8:
                data[8] = vector_index
        except Exception:
            pass
        return vector_index

    # --- Enrollment: add/update/remove person ---
    def _handle_put_person(user_id: str, user_name: str, user_image: str, resp_meta: Optional[dict] = None):
        try:
            uid = str(user_id).strip()
            uname = str(user_name).strip() if user_name is not None else ''
            img = str(user_image).strip()
            if not uid or not img or not os.path.exists(img):
                print(f"[ENROLL] invalid request uid={uid} img={img}")
                return
            # Generate embedding via TRT
            model = _get_trt_model()
            if model is None:
                print("[ENROLL] TRT model unavailable")
                return
            # Align to 112x112 RGB
            rgb112 = _align_arcface_from_image_path(img)
            if rgb112 is None:
                print(f"[ENROLL] failed to read/align image: {img}")
                return
            x = rgb112.astype(np.float32)
            x -= 127.5
            x /= 128.0
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, axis=0).astype(np.float32)
            inp = np.array(x, dtype=np.float32, order="C")
            out = model.infer(inp)[0]
            emb = np.asarray(out).reshape(1, -1).astype(np.float32)
            # L2 normalize for cosine/IP
            n = float(np.linalg.norm(emb)) + 1e-12
            emb = (emb / n).astype(np.float32)

            idx = _ensure_index()
            # Update label mapping metadata (persons) while keeping per-vector labels in sync
            labels_path = lbl_path
            index_path = idx_path
            if not labels_path:
                labels_path = str(recog_cfg.get('labels_path', '')).strip()
            if not index_path:
                index_path = str(recog_cfg.get('index_path', '')).strip()
            persons = {}
            try:
                if labels_path and os.path.exists(labels_path):
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f) or {}
                    persons = meta.get('persons', {}) if isinstance(meta, dict) else {}
                    if not isinstance(persons, dict):
                        persons = {}
            except Exception:
                persons = {}
            # Apply add/update/remove: if user_name empty -> remove mapping; else upsert name
            if uname:
                persons[uid] = {'name': uname}
            else:
                if uid in persons:
                    persons.pop(uid, None)

            # Remove existing vectors labeled with this uid, then add new embedding with uid label
            # Note: we do NOT delete person entirely unless uname is empty
            try:
                _ = idx.remove_label(uid)
            except Exception:
                pass
            idx.add([uid], emb)
            # Persist index and labels/persons
            if index_path and labels_path:
                try:
                    # Save index with per-vector labels list
                    idx.save(index_path, labels_path)
                    # Save persons mapping
                    meta = {}
                    try:
                        with open(labels_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f) or {}
                    except Exception:
                        meta = {}
                    meta['persons'] = persons
                    with open(labels_path, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    print(f"[ENROLL] saved uid={uid} name='{uname}' to index")
                except Exception as e:
                    print(f"[ENROLL] save failed: {e}")
        except Exception as e:
            print(f"[ENROLL] error: {e}")

    # --- Delete person (remove from index and persons map) ---
    def _handle_del_person(user_id: str, resp_meta: Optional[dict] = None):
        try:
            uid = str(user_id).strip()
            if not uid:
                return False
            idx = _ensure_index()
            removed = 0
            try:
                removed = idx.remove_label(uid)
            except Exception:
                removed = 0
            # Persist changes
            labels_path = lbl_path or str(recog_cfg.get('labels_path', '')).strip()
            index_path = idx_path or str(recog_cfg.get('index_path', '')).strip()
            if index_path and labels_path:
                try:
                    idx.save(index_path, labels_path)
                    # Remove from persons map as well
                    meta = {}
                    try:
                        if os.path.exists(labels_path):
                            with open(labels_path, 'r', encoding='utf-8') as f:
                                meta = json.load(f) or {}
                    except Exception:
                        meta = {}
                    persons = meta.get('persons', {}) if isinstance(meta, dict) else {}
                    if isinstance(persons, dict) and uid in persons:
                        persons.pop(uid, None)
                    meta['persons'] = persons
                    with open(labels_path, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    print(f"[ENROLL] removed uid={uid}, vectors_removed={removed}")
                except Exception as e:
                    print(f"[ENROLL] remove save failed: {e}")
        except Exception as e:
            print(f"[ENROLL] del error: {e}")

    # --- MQTT integration: listen for core_v2 requests to add/update sources and enroll persons ---
    def _parse_and_handle_request(obj):
        try:
            # Parse payload as semicolon-separated tokens or structured dict
            tokens = None
            if isinstance(obj, dict):
                if 'raw' in obj:
                    try:
                        payload_str = obj['raw'].decode('utf-8', errors='ignore')
                        tokens = [t.strip() for t in payload_str.split(';')]
                    except Exception:
                        tokens = None
                elif 'payload' in obj and isinstance(obj['payload'], str):
                    tokens = [t.strip() for t in obj['payload'].split(';')]
                elif 'cmd' in obj:
                    # Structured JSON
                    cmd = int(obj.get('cmd', 0))
                    if cmd == 25:  # enable service
                        cam_id = str(obj.get('cam_id', ''))
                        rtsp = str(obj.get('rtsp', ''))
                        face_enable = int(obj.get('face_enable', 0))
                        cmd_id = str(obj.get('cmd_id',''))
                        if face_enable == 1 and cam_id and rtsp:
                            GLib.idle_add(_add_or_update_source, cam_id, rtsp, {'cmd':25,'cam_id':cam_id,'cmd_id':cmd_id,'status':0})
                        elif face_enable == 0 and cam_id:
                            GLib.idle_add(_remove_source, cam_id, {'cmd':25,'cam_id':cam_id,'cmd_id':cmd_id,'status':0})
                    elif cmd == 26:  # disable service directly
                        cam_id = str(obj.get('cam_id', ''))
                        cmd_id = str(obj.get('cmd_id',''))
                        if cam_id:
                            GLib.idle_add(_remove_source, cam_id, {'cmd':26,'cam_id':cam_id,'cmd_id':cmd_id,'status':0})
                    elif cmd == 60:  # CMD_BOX_PUT_PERSON
                        uid = str(obj.get('user_id', ''))
                        uname = str(obj.get('user_name', ''))
                        uimg = str(obj.get('user_image', ''))
                        cmd_id = str(obj.get('cmd_id',''))
                        GLib.idle_add(_handle_put_person, uid, uname, uimg, {'cmd':60,'cmd_id':cmd_id,'status':2})
                    elif cmd == 61:  # CMD_BOX_DEL_PERSON
                        uid = str(obj.get('user_id', ''))
                        cmd_id = str(obj.get('cmd_id',''))
                        GLib.idle_add(_handle_del_person, uid, {'cmd':61,'cmd_id':cmd_id,'status':3})
                    return
            if not tokens or len(tokens) < 3:
                return
            def _tok(i, default=''):
                try:
                    return tokens[i]
                except Exception:
                    return default
            # xcommon.h indices
            REQ_IDX_CMD = 0
            REQ_IDX_CAMID = 1
            REQ_IDX_RTSP_URL = 2
            REQ_IDX_FACE_ENABLE = 15
            REQ_IDX_USER_ID = 2
            REQ_IDX_USER_NAME = 3
            REQ_IDX_USER_IMG_PATH = 4
            # Extract cmd_id (last non-empty token)
            cmd_id = ''
            for t in reversed(tokens):
                if t.strip():
                    cmd_id = t.strip()
                    break
            try:
                cmd = int(_tok(REQ_IDX_CMD, '0') or '0')
            except Exception:
                cmd = 0
            if cmd == 25:  # CMD_BOX_ENABLE_SERVICE
                cam_id = _tok(REQ_IDX_CAMID, '')
                rtsp = _tok(REQ_IDX_RTSP_URL, '')
                try:
                    face_enable = int(_tok(REQ_IDX_FACE_ENABLE, '0') or '0')
                except Exception:
                    face_enable = 0
                if face_enable == 1 and cam_id and rtsp:
                    print(f"[MQTT] Enable face_core for cam={cam_id}, rtsp={rtsp}")
                    GLib.idle_add(_add_or_update_source, cam_id, rtsp, {'cmd':25,'cam_id':cam_id,'cmd_id':cmd_id,'status':0})
                elif face_enable == 0 and cam_id:
                    GLib.idle_add(_remove_source, cam_id, {'cmd':25,'cam_id':cam_id,'cmd_id':cmd_id,'status':0})
                return
            if cmd == 26:  # CMD_BOX_DISABLE_SERVICE
                cam_id = _tok(REQ_IDX_CAMID, '')
                if cam_id:
                    print(f"[MQTT] Disable face_core for cam={cam_id}")
                    GLib.idle_add(_remove_source, cam_id, {'cmd':26,'cam_id':cam_id,'cmd_id':cmd_id,'status':0})
                return
            if cmd == 60:  # CMD_BOX_PUT_PERSON
                uid = _tok(REQ_IDX_USER_ID, '')
                uname = _tok(REQ_IDX_USER_NAME, '')
                uimg = _tok(REQ_IDX_USER_IMG_PATH, '')
                GLib.idle_add(_handle_put_person, uid, uname, uimg, {'cmd':60,'cmd_id':cmd_id,'status':2})
                return
            if cmd == 61:  # CMD_BOX_DEL_PERSON
                uid = _tok(REQ_IDX_USER_ID, '')
                GLib.idle_add(_handle_del_person, uid, {'cmd':61,'cmd_id':cmd_id,'status':3})
                return
        except Exception as e:
            try:
                print(f"[MQTT] parse error: {e}")
            except Exception:
                pass

    try:
        mqtt_cfg = cfg.get('mqtt', {}) if isinstance(cfg, dict) else {}
        host = str(mqtt_cfg.get('host', 'localhost'))
        port = int(mqtt_cfg.get('port', 1883))
        topic = str(mqtt_cfg.get('request_topic', '/local/core/v2/ai/request'))
        resp_topic = str(mqtt_cfg.get('response_topic', '/local/core/v2/ai/response'))
        _mqtt = MQTTListener(host=host, port=port, topic=topic, on_message=_parse_and_handle_request)
        _mqtt.start()
        print(f"[MQTT] listening on {host}:{port} topic {topic}")

        # Response helper: format "3;cmd;cam_or_zero;user_id_or_cam;cmd_id;status"
        def _publish_response(cmd: int, cam_id: str, user_id: str, cmd_id: str, status: int):
            try:
                # If cam_id not provided but user context exists, place '0' as second field like examples
                field_cam_slot = cam_id if cam_id else '0'
                field_user_slot = user_id if user_id else cam_id
                parts = [
                    '3',                     # RET_RESP
                    str(cmd),                # command
                    str(field_cam_slot),     # cam id or 0
                    str(field_user_slot or ''), # user id or cam id
                    str(cmd_id or ''),       # command id token tail
                    str(status)              # status code
                ]
                payload = ';'.join(parts)
                _mqtt.publish(resp_topic, payload)
            except Exception:
                pass

        mqtt_listener_ref = _mqtt  # keep reference to avoid GC
    except Exception as e:
        try:
            print(f"[MQTT] not started: {e}")
        except Exception:
            pass
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

