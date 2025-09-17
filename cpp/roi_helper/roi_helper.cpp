#include "roi_helper.h"
#include <gst/gst.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <cstdlib>
#include <cstring>

static inline int clampi(int v, int lo, int hi) {
  if (v < lo) return lo; if (v > hi) return hi; return v;
}

int roi_crop_bgr(uint64_t gst_buffer_ptr,
                 int batch_id,
                 int left,
                 int top,
                 int width,
                 int height,
                 int out_w,
                 int out_h,
                 uint8_t** out_buf,
                 int* out_size) {
  if (!gst_buffer_ptr || !out_buf || !out_size) return -1;
  GstBuffer* buffer = reinterpret_cast<GstBuffer*>(gst_buffer_ptr);
  NvBufSurface* surf = (NvBufSurface*) gst_mini_object_get_qdata(GST_MINI_OBJECT(buffer), g_quark_from_static_string("NvBufSurface"));
  if (!surf) return -2;
  if (batch_id < 0 || batch_id >= (int)surf->batchSize) return -3;

  NvBufSurfaceParams& sp = surf->surfaceList[batch_id];
  int W = sp.width;
  int H = sp.height;
  if (W <= 0 || H <= 0) return -4;

  int x0 = clampi(left, 0, W - 1);
  int y0 = clampi(top, 0, H - 1);
  int x1 = clampi(left + width, 0, W);
  int y1 = clampi(top + height, 0, H);
  int roi_w = x1 - x0;
  int roi_h = y1 - y0;
  if (roi_w <= 0 || roi_h <= 0) return -5;

  int dst_w = out_w > 0 ? out_w : roi_w;
  int dst_h = out_h > 0 ? out_h : roi_h;

  NvBufSurfTransformRect src_rect, dst_rect;
  src_rect.left = x0; src_rect.top = y0; src_rect.width = roi_w; src_rect.height = roi_h;
  dst_rect.left = 0; dst_rect.top = 0; dst_rect.width = dst_w; dst_rect.height = dst_h;

  NvBufSurface* dst_surf = nullptr;
  NvBufSurfaceCreateParams cparams;
  memset(&cparams, 0, sizeof(cparams));
  cparams.gpuId = surf->gpuId;
  cparams.width = dst_w;
  cparams.height = dst_h;
  cparams.layout = NVBUF_LAYOUT_PITCH;
  // Use RGBA (widely supported), we'll convert to BGR when copying out
  cparams.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
#ifndef PLATFORM_TEGRA
  cparams.memType = NVBUF_MEM_CUDA_UNIFIED;
#else
  cparams.memType = NVBUF_MEM_DEFAULT;
#endif
  if (NvBufSurfaceCreate(&dst_surf, 1, &cparams) != 0 || !dst_surf) {
    return -6;
  }

  NvBufSurfTransformParams tparams;
  memset(&tparams, 0, sizeof(tparams));
  tparams.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  tparams.transform_filter = NvBufSurfTransformInter_Default;
  tparams.src_rect = &src_rect;
  tparams.dst_rect = &dst_rect;
  NvBufSurfTransformConfigParams cfg_params;
  memset(&cfg_params, 0, sizeof(cfg_params));
  cfg_params.compute_mode = NvBufSurfTransformCompute_Default;
  cfg_params.gpu_id = surf->gpuId;
  cfg_params.cuda_stream = NULL;
  NvBufSurfTransformSetSessionParams(&cfg_params);

  if (NvBufSurfTransform(surf, dst_surf, &tparams) != 0) {
    NvBufSurfaceDestroy(dst_surf);
    return -8;
  }

  if (NvBufSurfaceMap(dst_surf, 0, 0, NVBUF_MAP_READ) != 0) {
    NvBufSurfaceDestroy(dst_surf);
    return -9;
  }
  NvBufSurfaceSyncForCpu(dst_surf, 0, 0);

  int pitch = dst_surf->surfaceList[0].pitch;
  unsigned char* ptr = (unsigned char*)dst_surf->surfaceList[0].mappedAddr.addr[0];
  int outsz = dst_w * dst_h * 3; // BGR bytes
  uint8_t* out = (uint8_t*)malloc(outsz);
  if (!out) {
    NvBufSurfaceUnMap(dst_surf, 0, 0);
    NvBufSurfaceDestroy(dst_surf);
    return -10;
  }

  // Convert RGBA -> BGR row by row (handle pitch)
  for (int y = 0; y < dst_h; ++y) {
    unsigned char* src_row = ptr + y * pitch;
    uint8_t* dst_row = out + y * dst_w * 3;
    for (int x = 0; x < dst_w; ++x) {
      unsigned char* s = src_row + 4 * x; // RGBA
      uint8_t* d = dst_row + 3 * x;       // BGR
      d[0] = s[2]; // B
      d[1] = s[1]; // G
      d[2] = s[0]; // R
    }
  }

  NvBufSurfaceUnMap(dst_surf, 0, 0);
  NvBufSurfaceDestroy(dst_surf);

  *out_buf = out;
  *out_size = outsz;
  return 0;
}

void roi_free(uint8_t* buf) {
  if (buf) free(buf);
}
