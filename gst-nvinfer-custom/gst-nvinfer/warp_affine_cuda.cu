// warp_affine_cuda.cu
// CUDA warpAffine (bilinear, border constant) for interleaved uint8 images (3 or 4 channels) with pitch

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ inline uint8_t read_pixel(const uint8_t* base, int x, int y, int pitch, int c, int channels, int width, int height, uint8_t border) {
    if (x < 0 || y < 0 || x >= width || y >= height) return border;
    const uint8_t* row = base + y * pitch;
    return row[x * channels + c];
}

template<int CHANNELS>
__global__ void warp_affine_bilinear_kernel(const uint8_t* __restrict__ src, int src_w, int src_h, int src_pitch,
                                            uint8_t* __restrict__ dst, int dst_w, int dst_h, int dst_pitch,
                                            // Pass matrix by value to avoid device memory lifetime issues
                                            float a, float b, float c, float d, float e, float f,
                                            uint8_t border_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float xs = a * x + b * y + c;
    float ys = d * x + e * y + f;

    // If mapping is invalid or far outside, write border and return
    if (!isfinite(xs) || !isfinite(ys) || xs < -1.0f || ys < -1.0f || xs > (float)src_w || ys > (float)src_h) {
        uint8_t* out_row_b = dst + y * dst_pitch + x * CHANNELS;
        #pragma unroll
        for (int ch = 0; ch < CHANNELS; ++ch) out_row_b[ch] = border_value;
        return;
    }

    int x0 = static_cast<int>(floorf(xs));
    int y0 = static_cast<int>(floorf(ys));
    float ax = xs - x0;
    float ay = ys - y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    uint8_t* out_row = dst + y * dst_pitch + x * CHANNELS;
    #pragma unroll
    for (int ch = 0; ch < CHANNELS; ++ch) {
        float v00 = (float)read_pixel(src, x0, y0, src_pitch, ch, CHANNELS, src_w, src_h, border_value);
        float v01 = (float)read_pixel(src, x1, y0, src_pitch, ch, CHANNELS, src_w, src_h, border_value);
        float v10 = (float)read_pixel(src, x0, y1, src_pitch, ch, CHANNELS, src_w, src_h, border_value);
        float v11 = (float)read_pixel(src, x1, y1, src_pitch, ch, CHANNELS, src_w, src_h, border_value);
        float v0 = v00 + ax * (v01 - v00);
        float v1 = v10 + ax * (v11 - v10);
        float vf = v0 + ay * (v1 - v0);
        int vi = (int)roundf(vf);
        if (vi < 0) vi = 0; else if (vi > 255) vi = 255;
        out_row[ch] = (uint8_t)vi;
    }
}

extern "C" void warp_affine_bilinear_u8(const uint8_t* src, int src_w, int src_h, int src_pitch,
                                         uint8_t* dst, int dst_w, int dst_h, int dst_pitch,
                                         const float* invM2x3 /* host ptr */, int channels,
                                         uint8_t border_value, cudaStream_t stream) {
    // Read matrix on host and pass by value to kernel
    float a = invM2x3[0];
    float b = invM2x3[1];
    float c = invM2x3[2];
    float d = invM2x3[3];
    float e = invM2x3[4];
    float f = invM2x3[5];

    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    if (channels == 3) {
        warp_affine_bilinear_kernel<3><<<grid, block, 0, stream>>>(src, src_w, src_h, src_pitch,
                                                                   dst, dst_w, dst_h, dst_pitch,
                                                                   a, b, c, d, e, f,
                                                                   border_value);
    } else {
        warp_affine_bilinear_kernel<4><<<grid, block, 0, stream>>>(src, src_w, src_h, src_pitch,
                                                                   dst, dst_w, dst_h, dst_pitch,
                                                                   a, b, c, d, e, f,
                                                                   border_value);
    }
}
