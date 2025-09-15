// svd_cuda_full.cpp
// cuSOLVER SVD that returns U, S, VT on host; helper to compute R = U*D*VT for Umeyama

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <vector>

extern "C" void svd_full_cuda(const float* h_A, int rows, int cols,
                               std::vector<float>& U, std::vector<float>& S, std::vector<float>& VT) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    int m = rows, n = cols;
    int lda = m, ldu = m, ldvt = n;
    int min_mn = (m < n) ? m : n;

    float *d_A=nullptr, *d_S=nullptr, *d_U=nullptr, *d_VT=nullptr;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_S, min_mn * sizeof(float));
    cudaMalloc(&d_U, ldu * m * sizeof(float));
    cudaMalloc(&d_VT, ldvt * n * sizeof(float));

    int lwork = 0; cusolverDnSgesvd_bufferSize(handle, m, n, &lwork);
    float* d_work=nullptr; cudaMalloc(&d_work, lwork * sizeof(float));
    int* devInfo=nullptr; cudaMalloc(&devInfo, sizeof(int));
    char jobu='A', jobvt='A';
    cusolverDnSgesvd(handle, jobu, jobvt, m, n, d_A, lda, d_S, d_U, ldu, d_VT, ldvt, d_work, lwork, nullptr, devInfo);
    int info=0; cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0) {
        cudaFree(d_A); cudaFree(d_S); cudaFree(d_U); cudaFree(d_VT); cudaFree(d_work); cudaFree(devInfo);
        cusolverDnDestroy(handle);
        throw std::runtime_error("cusolverDnSgesvd failed");
    }

    U.resize(ldu * m); S.resize(min_mn); VT.resize(ldvt * n);
    cudaMemcpy(U.data(), d_U, ldu * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S.data(), d_S, min_mn * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(VT.data(), d_VT, ldvt * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_S); cudaFree(d_U); cudaFree(d_VT); cudaFree(d_work); cudaFree(devInfo);
    cusolverDnDestroy(handle);
}

extern "C" void compute_rotation_from_svd_umeyama(const std::vector<float>& U, const std::vector<float>& S, const std::vector<float>& VT,
                                                    int dim, float* R_out /* dim x dim row-major */) {
    // U (dim x dim), VT (dim x dim), S (dim)
    // D = diag(1,..,1, det(U*VT) >= 0 ? 1 : -1)
    // R = U * D * VT
    // Compute sign using simple determinant sign of U*VT via SVD property for small dim (2 or 3)
    // For robustness, compute M = U*VT and det(M) directly for dim<=3
    float M[9] = {0};
    for (int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            float acc=0.f;
            for(int k=0;k<dim;k++) acc += U[i*dim+k]*VT[j*dim+k]; // note VT is already transposed; U*VT
            M[i*dim+j]=acc;
        }
    }
    float det=0.f;
    if (dim==2) {
        det = M[0]*M[3]-M[1]*M[2];
    } else if (dim==3) {
        det = M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]);
    } else {
        // fallback sign
        det = 1.f;
    }
    float D[9] = {0};
    for (int i=0;i<dim;i++) D[i*dim+i]=1.f;
    if (det < 0) D[(dim-1)*dim + (dim-1)] = -1.f;
    float UD[9] = {0};
    for (int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            float acc=0.f; for(int k=0;k<dim;k++) acc += U[i*dim+k]*D[k*dim+j];
            UD[i*dim+j]=acc;
        }
    }
    for (int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            float acc=0.f; for(int k=0;k<dim;k++) acc += UD[i*dim+k]*VT[k*dim+j];
            R_out[i*dim+j]=acc;
        }
    }
}
