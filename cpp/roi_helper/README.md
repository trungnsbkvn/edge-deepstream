ROI Helper

A small DeepStream-compatible C++ helper to crop object ROI from NvBufSurface to BGR and return bytes to Python.

Build:
- Requires DeepStream SDK headers/libs on Jetson (nvvideoconvert/NvBufSurface/NvBufSurfTransform).
- Uses cmake.

Install:
- Produces libroi_helper.so under build/. Copy to project root or add to LD_LIBRARY_PATH.
