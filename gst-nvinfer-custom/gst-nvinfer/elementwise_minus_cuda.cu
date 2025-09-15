// CUDA kernel for ElementwiseMinus: output[i, j] = A[i, j] - B[0, j]
// A: (rows x cols), B: (1 x cols), output: (rows x cols)
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void elementwise_minus_kernel(const float* A, const float* B, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int i = idx / cols;
    int j = idx % cols;
    output[idx] = A[idx] - B[j];
}

extern "C" void elementwise_minus_cuda(const float* d_A, const float* d_B, float* d_output, int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    elementwise_minus_kernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_output, rows, cols);
}
