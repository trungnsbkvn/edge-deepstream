#!/bin/bash
set -euo pipefail

export DS_CONFIG_PATH=config/config_pipeline_rtsp.toml
export DS_RTSP_LATENCY=150
export DS_RTSP_TCP=1
export DS_RTSP_DROP_ON_LATENCY=1
export DS_RTSP_RETRANS=0
export DS_REALTIME_DROP=1
export PGIE_MIN_CONF=0.5
export PGIE_DEBUG_DETS=1

echo "Starting pipeline with RTSP source config: $DS_CONFIG_PATH" >&2
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 main.py