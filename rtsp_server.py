#!/usr/bin/env python3
import os
import sys
import time
import gi

gi.require_version('Gst', '1.0')
try:
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import Gst, GObject, GLib, GstRtspServer  # type: ignore
except ValueError:
    from gi.repository import Gst, GLib  # type: ignore
    GstRtspServer = None  # type: ignore
    print("[ERROR] GstRtspServer GObject introspection bindings not available.\n"
          "Install the GStreamer RTSP server dev packages, e.g.:\n"
          "  sudo apt-get update && sudo apt-get install -y libgstrtspserver-1.0-dev gir1.2-gst-rtsp-server-1.0\n"
          "Or use the mediamtx fallback instructions below.")

"""
Simple RTSP server that streams the local MP4 test video as H264.
Usage:
  python3 rtsp_server.py [video_path] [port] [mount]
Defaults:
  video_path = data/media/friends_s1e1_cut.mp4
  port       = 8554
  mount      = /test

Resulting RTSP URL: rtsp://<host>:<port><mount>

If the MP4 is not already H264, GStreamer will transcode. For performance on Jetson,
keep sources already in H264.
"""
def _select_encoder():
    if os.environ.get('USE_X264', '0') == '1':
        return 'x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=30'
    # Prefer Jetson / NV hardware encoder
    return 'nvv4l2h264enc preset-level=1 iframeinterval=30 insert-sps-pps=1 idrinterval=30 bitrate=4000000 ! h264parse config-interval=1'


LOOP_LAUNCH = None  # debug visibility


def main():
    Gst.init(None)
    if GstRtspServer is None:
        print("\nFALLBACK OPTIONS:\n"
              "1) Install the missing packages (recommended).\n"
              "2) Use mediamtx to serve the file (see earlier instructions).\n"
              "3) Temporarily point your pipeline back to file:// source.\n")
        return 2

    video_path = sys.argv[1] if len(sys.argv) > 1 else 'data/media/friends_s1e1_cut.mp4'
    port = sys.argv[2] if len(sys.argv) > 2 else '8554'
    mount = sys.argv[3] if len(sys.argv) > 3 else '/test'
    target_fps = None
    try:
        if 'RTSP_FPS' in os.environ:
            target_fps = int(os.environ['RTSP_FPS'])
    except Exception:
        target_fps = None

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return 1

    server = GstRtspServer.RTSPServer()
    server.props.service = port
    mounts = server.get_mount_points()

    # Build launch string
    def build_launch():
        encoder = _select_encoder()
        direct = (video_path.lower().endswith('.mp4') and os.environ.get('FORCE_TRANSCODE','0') != '1')
        fps_part = ''
        if target_fps:
            fps_part = f"videorate ! video/x-raw,framerate={target_fps}/1 ! "
        if direct:
            # Direct remux path (no re-encode) – only handle first video stream; audio pads will remain unlinked (ignored)
            # Use parse -> pay. If hardware clients require periodic SPS/PPS, config-interval=1 already set.
            return (
                f"filesrc location={video_path} ! qtdemux name=demux demux. ! queue leaky=2 max-size-buffers=60 ! h264parse config-interval=1 "
                f"! rtph264pay name=pay0 pt=96 config-interval=1"
            )
        else:
            return (
                f"filesrc location={video_path} ! decodebin name=dec dec. ! queue leaky=2 max-size-buffers=30 ! videoconvert ! {fps_part}{encoder} "
                f"! rtph264pay name=pay0 pt=96 config-interval=1"
            )

    def _is_direct():
        return (video_path.lower().endswith('.mp4') and os.environ.get('FORCE_TRANSCODE','0') != '1')

    launch = build_launch()
    global LOOP_LAUNCH
    LOOP_LAUNCH = launch
    print(f"[RTSP] Launch pipeline: {launch}")

    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_shared(True)
    factory.set_launch(f"( {launch} )")

    def _on_media_configure(factory, media):  # noqa
        element = media.get_element()
        # Poll duration and position; when near end, perform seek to loop.
        # This avoids adding a bus watch (which failed earlier on this system).
        loop_margin_ns_default = 2 * 1_000_000_000  # 2 seconds
        poll_interval_ms = 500
        state = {
            'looped': False,
            'last_seek_ts': 0.0,
            'last_position': -1,
            'stalled_loops': 0,
            'switch_done': False,
        }

        def _poll_loop():
            try:
                # Require PLAYING
                if element.get_state(0)[1] != Gst.State.PLAYING:
                    return True
                success_dur, duration = element.query_duration(Gst.Format.TIME)
                success_pos, position = element.query_position(Gst.Format.TIME)
                if not (success_dur and success_pos and duration > 0 and position >= 0):
                    return True
                # Adaptive margin: 5% of duration capped at 2s
                adaptive_margin = int(min(loop_margin_ns_default, max(duration * 0.05, 500_000_000)))
                near_end = (duration - position) <= adaptive_margin
                # Reset loop flag once we're safely away from end (< 1 second)
                if state['looped'] and position < 1_000_000_000:
                    state['looped'] = False
                if near_end and not state['looped']:
                    print(f"[RTSP] Looping (poll) pos={position/1e9:.2f}s dur={duration/1e9:.2f}s margin={adaptive_margin/1e9:.2f}s")
                    seek_ok = element.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
                    if not seek_ok:
                        print('[RTSP] seek_simple failed; attempting state cycle')
                        element.set_state(Gst.State.READY)
                        element.set_state(Gst.State.PLAYING)
                    state['looped'] = True
                    state['last_seek_ts'] = time.time()
                    state['last_position'] = 0
                # Detect failed looping (position not resetting after seek attempts)
                if state['looped']:
                    # If we are still near end for >3 polls and position did not drop, classify as stall
                    if near_end and position == state.get('last_position', -1):
                        state['stalled_loops'] += 1
                    else:
                        state['stalled_loops'] = 0
                    # After several failed attempts, restart element
                    if state['stalled_loops'] >= 4:
                        print('[RTSP] Loop stall detected; restarting pipeline element')
                        element.set_state(Gst.State.NULL)
                        element.set_state(Gst.State.READY)
                        element.set_state(Gst.State.PLAYING)
                        state['stalled_loops'] = 0
                        state['looped'] = False
                        # Auto-switch to transcode path if direct remux failing repeatedly
                        if _is_direct() and not state['switch_done']:
                            print('[RTSP] Auto-switching to transcode path due to looping stall')
                            os.environ['FORCE_TRANSCODE'] = '1'
                            # Rebuild launch string
                            new_launch = build_launch()
                            print(f"[RTSP] New launch: {new_launch}")
                            # Replace factory launch (new clients)
                            factory.set_launch(f"( {new_launch} )")
                            # Attempt to re-prime existing media element states
                            element.set_state(Gst.State.NULL)
                            element.set_state(Gst.State.READY)
                            element.set_state(Gst.State.PLAYING)
                            state['switch_done'] = True
                state['last_position'] = position
                return True
            except Exception:
                return True
        GLib.timeout_add(poll_interval_ms, _poll_loop)
    factory.connect('media-configure', _on_media_configure)

    mounts.add_factory(mount, factory)

    server.attach(None)
    print(f"RTSP server streaming (looping) {video_path}\n  URL: rtsp://0.0.0.0:{port}{mount}\nPress Ctrl+C to stop.")

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Stopping RTSP server...")
        loop.quit()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Stopping RTSP server...")
        loop.quit()

if __name__ == '__main__':
    sys.exit(main() or 0)
