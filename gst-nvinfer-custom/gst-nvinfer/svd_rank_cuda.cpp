// C++/CUDA wrapper for SVD and matrix rank using cuSOLVER
// This is a minimal example for small matrices (e.g., 2x2, 3x3)
// For production, add error checking and workspace management
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <stdexcept>

extern "C" int matrix_rank_svd_cuda(const float* h_A, int rows, int cols, float tol = 1e-4) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    float* d_A = nullptr;
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int m = rows, n = cols;
    int lda = m;
    int ldu = m, ldvt = n;
    int min_mn = (m < n) ? m : n;
    std::vector<float> S(min_mn);
    float* d_S = nullptr;
    cudaMalloc(&d_S, min_mn * sizeof(float));
    float* d_U = nullptr; cudaMalloc(&d_U, ldu * m * sizeof(float));
    float* d_VT = nullptr; cudaMalloc(&d_VT, ldvt * n * sizeof(float));
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(handle, m, n, &lwork);
    float* d_work = nullptr; cudaMalloc(&d_work, lwork * sizeof(float));
    int* devInfo = nullptr; cudaMalloc(&devInfo, sizeof(int));

    char jobu = 'A', jobvt = 'A';
    cusolverDnSgesvd(handle, jobu, jobvt, m, n, d_A, lda, d_S, d_U, ldu, d_VT, ldvt, d_work, lwork, nullptr, devInfo);
    cudaMemcpy(S.data(), d_S, min_mn * sizeof(float), cudaMemcpyDeviceToHost);

    int info = 0;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_S); cudaFree(d_U); cudaFree(d_VT); cudaFree(d_work); cudaFree(devInfo);
    cusolverDnDestroy(handle);
    if (info != 0) throw std::runtime_error("SVD failed");
    int rank = 0;
    for (int i = 0; i < min_mn; ++i) {
        if (S[i] > tol) ++rank;
    }
    return rank;
}
