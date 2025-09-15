// CUDA kernel for computing mean along axis 0 (column-wise mean)
// Input: src (num_rows x num_cols), row-major float32
// Output: mean (num_cols)

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void mean_axis0_kernel(const float* src, float* mean, int num_rows, int num_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= num_cols) return;
    float sum = 0.0f;
    for (int row = 0; row < num_rows; ++row) {
        sum += src[row * num_cols + col];
    }
    mean[col] = sum / num_rows;
}

// Host wrapper for launching the kernel
extern "C" void mean_axis0_cuda(const float* d_src, float* d_mean, int num_rows, int num_cols, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_cols + threads - 1) / threads;
    mean_axis0_kernel<<<blocks, threads, 0, stream>>>(d_src, d_mean, num_rows, num_cols);
}
