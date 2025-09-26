#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Crop an ROI from the NvBufSurface inside a DeepStream GstBuffer into BGR image bytes.
// params:
//  gst_buffer: pointer to GstBuffer (as uint64 cast from Python)
//  batch_id: frame_meta.batch_id (index in NvBufSurface)
//  left, top, width, height: ROI in pixels
//  out_w, out_h: desired output size; if 0, keep original ROI size
//  out_buf: pointer to pointer receiving malloc'd BGR bytes (size = out_w*out_h*3)
//  out_size: number of bytes in out_buf
// returns 0 on success, non-zero on error.
int roi_crop_bgr(uint64_t gst_buffer,
                 int batch_id,
                 int left,
                 int top,
                 int width,
                 int height,
                 int out_w,
                 int out_h,
                 uint8_t** out_buf,
                 int* out_size);

// Free buffer allocated by roi_crop_bgr
void roi_free(uint8_t* buf);

#ifdef __cplusplus
}
#endif
