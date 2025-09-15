// matrix_ops_cuda_3x3.cu
// CUDA kernels and wrappers for 3x3 matrix ops: multiplication, determinant, transpose
// Only for 3x3 float matrices (row-major)

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

// Matrix multiplication: C = A * B (row-major, 3x3)
__global__ void matmul3x3_kernel(const float* A, const float* B, float* C) {
    int i = threadIdx.x;
    if (i < 9) {
        int row = i / 3, col = i % 3;
        C[i] = 0.0f;
        for (int k = 0; k < 3; ++k) {
            C[i] += A[row * 3 + k] * B[k * 3 + col];
        }
    }
}

void matmul3x3_cuda(const float* h_A, const float* h_B, float* h_C) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 9 * sizeof(float));
    cudaMalloc(&d_B, 9 * sizeof(float));
    cudaMalloc(&d_C, 9 * sizeof(float));
    cudaMemcpy(d_A, h_A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 9 * sizeof(float), cudaMemcpyHostToDevice);
    matmul3x3_kernel<<<1, 9>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Determinant for 3x3 matrix
__global__ void det3x3_kernel(const float* A, float* det) {
    *det = A[0]*(A[4]*A[8] - A[5]*A[7]) - A[1]*(A[3]*A[8] - A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
}

void det3x3_cuda(const float* h_A, float* h_det) {
    float *d_A, *d_det;
    cudaMalloc(&d_A, 9 * sizeof(float));
    cudaMalloc(&d_det, sizeof(float));
    cudaMemcpy(d_A, h_A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    det3x3_kernel<<<1, 1>>>(d_A, d_det);
    cudaMemcpy(h_det, d_det, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_det);
}

// Transpose 3x3
__global__ void transpose3x3_kernel(const float* A, float* At) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            At[j * 3 + i] = A[i * 3 + j];
}

void transpose3x3_cuda(const float* h_A, float* h_At) {
    float *d_A, *d_At;
    cudaMalloc(&d_A, 9 * sizeof(float));
    cudaMalloc(&d_At, 9 * sizeof(float));
    cudaMemcpy(d_A, h_A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    transpose3x3_kernel<<<1, 1>>>(d_A, d_At);
    cudaMemcpy(h_At, d_At, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_At);
}

// TODO: Add cuSOLVER SVD wrapper for 3x3 if needed

} // extern "C"
