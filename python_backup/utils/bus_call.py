import gi
import sys
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
from utils.env import _env_bool
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        # Allow keeping loop alive if all dynamic sources were removed intentionally.
        quit_on_empty = bool(_env_bool('DS_QUIT_ON_EMPTY', False))
        if quit_on_empty:
            sys.stdout.write("End-of-stream (quit_on_empty=1)\n")
            loop.quit()
        else:
            sys.stdout.write("End-of-stream (ignored; waiting for dynamic sources)\n")
        return True
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        err_str = str(err).lower()
        debug_str = str(debug).lower()
        # Don't quit on RTSP errors - let the pipeline continue
        if ("rtspsrc" in err_str or "rtsp" in err_str or 
            "rtspsrc" in debug_str or "rtsp" in debug_str or
            "option not supported" in err_str or "option not supported" in debug_str):
            sys.stderr.write("RTSP Error (non-fatal): %s: %s\n" % (err, debug))
        else:
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            loop.quit()
    return True
