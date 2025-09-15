// matrix_ops_cuda.cu
// CUDA kernels and wrappers for small matrix ops: multiplication, determinant, transpose, SVD (cuSOLVER)
// Only for 2x2 and 3x3 float matrices (row-major)

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>

extern "C" {

// Matrix multiplication: C = A * B (row-major, n x n)
__global__ void matmul2x2_kernel(const float* A, const float* B, float* C) {
    int i = threadIdx.x;
    if (i < 4) {
        int row = i / 2, col = i % 2;
        C[i] = A[row * 2 + 0] * B[0 * 2 + col] + A[row * 2 + 1] * B[1 * 2 + col];
    }
}

void matmul2x2_cuda(const float* h_A, const float* h_B, float* h_C) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 4 * sizeof(float));
    cudaMalloc(&d_B, 4 * sizeof(float));
    cudaMalloc(&d_C, 4 * sizeof(float));
    cudaMemcpy(d_A, h_A, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 4 * sizeof(float), cudaMemcpyHostToDevice);
    matmul2x2_kernel<<<1, 4>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Determinant for 2x2 matrix
__global__ void det2x2_kernel(const float* A, float* det) {
    *det = A[0] * A[3] - A[1] * A[2];
}

void det2x2_cuda(const float* h_A, float* h_det) {
    float *d_A, *d_det;
    cudaMalloc(&d_A, 4 * sizeof(float));
    cudaMalloc(&d_det, sizeof(float));
    cudaMemcpy(d_A, h_A, 4 * sizeof(float), cudaMemcpyHostToDevice);
    det2x2_kernel<<<1, 1>>>(d_A, d_det);
    cudaMemcpy(h_det, d_det, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_det);
}

// Transpose 2x2
__global__ void transpose2x2_kernel(const float* A, float* At) {
    At[0] = A[0]; At[1] = A[2];
    At[2] = A[1]; At[3] = A[3];
}

void transpose2x2_cuda(const float* h_A, float* h_At) {
    float *d_A, *d_At;
    cudaMalloc(&d_A, 4 * sizeof(float));
    cudaMalloc(&d_At, 4 * sizeof(float));
    cudaMemcpy(d_A, h_A, 4 * sizeof(float), cudaMemcpyHostToDevice);
    transpose2x2_kernel<<<1, 1>>>(d_A, d_At);
    cudaMemcpy(h_At, d_At, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_At);
}

// TODO: Add 3x3 versions and cuSOLVER SVD wrappers as needed

} // extern "C"
