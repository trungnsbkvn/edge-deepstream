// CUDA kernel for variance along axis 0 (column-wise variance)
// Input: src (num_rows x num_cols), row-major float32
// mean (num_cols) must be precomputed
// Output: var (num_cols)
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void var_axis0_kernel(const float* src, const float* mean, float* var, int num_rows, int num_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= num_cols) return;
    float sum = 0.0f;
    for (int row = 0; row < num_rows; ++row) {
        float diff = src[row * num_cols + col] - mean[col];
        sum += diff * diff;
    }
    var[col] = sum / num_rows;
}

extern "C" void var_axis0_cuda(const float* d_src, const float* d_mean, float* d_var, int num_rows, int num_cols, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_cols + threads - 1) / threads;
    var_axis0_kernel<<<blocks, threads, 0, stream>>>(d_src, d_mean, d_var, num_rows, num_cols);
}
