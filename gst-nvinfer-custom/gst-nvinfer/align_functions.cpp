// Ensure wrappers are defined inside alignnamespace and properly qualified
#include "align_functions.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

// Global C-linkage declaration for cuSOLVER SVD wrapper
extern "C" void svd_full_cuda(const float* h_A, int rows, int cols,
                               std::vector<float>& U, std::vector<float>& S, std::vector<float>& VT);

// Declaration of CUDA kernel wrapper
extern "C" void mean_axis0_cuda(const float* d_src, float* d_mean, int num_rows, int num_cols, cudaStream_t stream);
// GPU-accelerated mean along axis 0 (column-wise mean)
// h_src: host pointer to input matrix (row-major, num_rows x num_cols)
// h_mean: host pointer to output mean (num_cols)
// Both must be float32
void alignnamespace::Aligner::MeanAxis0CUDA(const float* h_src, float* h_mean, int num_rows, int num_cols, cudaStream_t stream) {
    float *d_src = nullptr, *d_mean = nullptr;
    size_t src_bytes = num_rows * num_cols * sizeof(float);
    size_t mean_bytes = num_cols * sizeof(float);
    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_mean, mean_bytes);
    cudaMemcpyAsync(d_src, h_src, src_bytes, cudaMemcpyHostToDevice, stream);
    mean_axis0_cuda(d_src, d_mean, num_rows, num_cols, stream);
    cudaMemcpyAsync(h_mean, d_mean, mean_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_src);
    cudaFree(d_mean);
}

// CUDA matrix ops declarations (2x2/3x3)
extern "C" void matmul2x2_cuda(const float* h_A, const float* h_B, float* h_C);
extern "C" void det2x2_cuda(const float* h_A, float* h_det);
extern "C" void transpose2x2_cuda(const float* h_A, float* h_At);
extern "C" void matmul3x3_cuda(const float* h_A, const float* h_B, float* h_C);
extern "C" void det3x3_cuda(const float* h_A, float* h_det);
extern "C" void transpose3x3_cuda(const float* h_A, float* h_At);

// Wrapper implementations qualified with namespace
void alignnamespace::Aligner::MatMul2x2CUDA(const float* h_A, const float* h_B, float* h_C) { matmul2x2_cuda(h_A, h_B, h_C); }
void alignnamespace::Aligner::Det2x2CUDA(const float* h_A, float* h_det) { det2x2_cuda(h_A, h_det); }
void alignnamespace::Aligner::Transpose2x2CUDA(const float* h_A, float* h_At) { transpose2x2_cuda(h_A, h_At); }
void alignnamespace::Aligner::MatMul3x3CUDA(const float* h_A, const float* h_B, float* h_C) { matmul3x3_cuda(h_A, h_B, h_C); }
void alignnamespace::Aligner::Det3x3CUDA(const float* h_A, float* h_det) { det3x3_cuda(h_A, h_det); }
void alignnamespace::Aligner::Transpose3x3CUDA(const float* h_A, float* h_At) { transpose3x3_cuda(h_A, h_At); }

// SVD-based matrix rank
extern "C" int matrix_rank_svd_cuda(const float* h_A, int rows, int cols, float tol);
int alignnamespace::Aligner::MatrixRankSVD_CUDA(const float* h_A, int rows, int cols, float tol) {
    return matrix_rank_svd_cuda(h_A, rows, cols, tol);
}

// Var and elementwise minus wrappers
extern "C" void var_axis0_cuda(const float* d_src, const float* d_mean, float* d_var, int num_rows, int num_cols, cudaStream_t stream);
void alignnamespace::Aligner::VarAxis0CUDA(const float* h_src, const float* h_mean, float* h_var, int num_rows, int num_cols, cudaStream_t stream) {
    float *d_src = nullptr, *d_mean = nullptr, *d_var = nullptr;
    size_t src_bytes = num_rows * num_cols * sizeof(float);
    size_t mean_bytes = num_cols * sizeof(float);
    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_mean, mean_bytes);
    cudaMalloc(&d_var, mean_bytes);
    cudaMemcpyAsync(d_src, h_src, src_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mean, h_mean, mean_bytes, cudaMemcpyHostToDevice, stream);
    var_axis0_cuda(d_src, d_mean, d_var, num_rows, num_cols, stream);
    cudaMemcpyAsync(h_var, d_var, mean_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_src);
    cudaFree(d_mean);
    cudaFree(d_var);
}

extern "C" void elementwise_minus_cuda(const float* d_A, const float* d_B, float* d_output, int rows, int cols, cudaStream_t stream);
void alignnamespace::Aligner::ElementwiseMinusCUDA(const float* h_A, const float* h_B, float* h_output, int rows, int cols, cudaStream_t stream) {
    float *d_A = nullptr, *d_B = nullptr, *d_output = nullptr;
    size_t A_bytes = rows * cols * sizeof(float);
    size_t B_bytes = cols * sizeof(float);
    cudaMalloc(&d_A, A_bytes);
    cudaMalloc(&d_B, B_bytes);
    cudaMalloc(&d_output, A_bytes);
    cudaMemcpyAsync(d_A, h_A, A_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, B_bytes, cudaMemcpyHostToDevice, stream);
    elementwise_minus_cuda(d_A, d_B, d_output, rows, cols, stream);
    cudaMemcpyAsync(h_output, d_output, A_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);
}

// warp affine CUDA wrapper
extern "C" void warp_affine_bilinear_u8(const uint8_t* src, int src_w, int src_h, int src_pitch,
                                         uint8_t* dst, int dst_w, int dst_h, int dst_pitch,
                                         const float* invM2x3, int channels, uint8_t border_value, cudaStream_t stream);
void alignnamespace::Aligner::WarpAffineU8CUDA(const uint8_t* d_src, int src_w, int src_h, int src_pitch,
                                               uint8_t* d_dst, int dst_w, int dst_h, int dst_pitch,
                                               const float* invM2x3, int channels, uint8_t border_value, cudaStream_t stream) {
    warp_affine_bilinear_u8(d_src, src_w, src_h, src_pitch, d_dst, dst_w, dst_h, dst_pitch, invM2x3, channels, border_value, stream);
}

// default_array use the norm landmarks arcface_src from
// https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
float standard_face[5][2] = {  
            {38.2946f, 51.6963f},
            {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f},
            {70.7299f, 92.2041f}
        };
// FFHQ face with size 512x512
float standard_face_ffhq[5][2] = {  
            {192.98138, 239.94708},
            {318.90277, 240.1936},
            {256.63416, 314.01935},
            {201.26117, 371.41043},
            {313.08905, 371.15118}
        };
// standard car plate
float standard_plate[4][2] = {
            {0.0f, 0.0f},
			{94.0f, 0.0f},
			{0.0f, 24.0f},
			{94.0f, 24.0f}
		};

float standard_plate_lpr3[4][2] = {
            {0.0f, 0.0f},
			{168.0f, 0.0f},
			{0.0f, 48.0f},
			{168.0f, 48.0f}
		};

namespace alignnamespace {
class Aligner::Impl {
public:
	cv::Mat Align(const cv::Mat& dst, int model_type);
    bool validLmks(float *landmarks);


private:
    cv::Mat MeanAxis0(const cv::Mat &src);
	cv::Mat ElementwiseMinus(const cv::Mat &A, const cv::Mat &B);
	cv::Mat VarAxis0(const cv::Mat &src);
	int MatrixRank(cv::Mat M);
	cv::Mat SimilarTransform(const cv::Mat& src, const cv::Mat& dst);
	
};


Aligner::Aligner() {
	impl_ = new Impl();
	
}

Aligner::~Aligner() {
	if (impl_) {
		delete impl_;
	}
}

cv::Mat Aligner::Align(const cv::Mat & dst, int model_type) {
	return impl_->Align(dst, model_type);
}

bool Aligner::validLmks(float *landmarks) {
    return impl_->validLmks(landmarks);
}


cv::Mat Aligner::Impl::Align(const cv::Mat & dst, int model_type) {
    if (model_type == 1) {
        cv::Mat src(5,2,CV_32FC1, standard_face);
        memcpy(src.data, standard_face, 2 * 5 * sizeof(float));
        cv::Mat M= SimilarTransform(dst, src);
        return M;
    }else if(model_type == 2) {
        cv::Mat src(4,2,CV_32FC1, standard_plate);
        memcpy(src.data, standard_plate, 2 * 4 * sizeof(float));
        cv::Mat M= cv::getPerspectiveTransform(dst, src);
        // cv::Mat M= SimilarTransform(dst, src);
        // std::cout << "start align plate." << std::endl;
        // std::cout << "end align face." << std::endl;
        return M;
    } else if (model_type == 3) {
        cv::Mat src(4,2,CV_32FC1, standard_plate_lpr3);
        memcpy(src.data, standard_plate_lpr3, 2 * 4 * sizeof(float));
        // std::cout<<(dst.checkVector(2, CV_32F) == 4)<<" "<<(src.checkVector(2, CV_32F) == 4)<<std::endl;
        cv::Mat M= cv::getPerspectiveTransform(dst, src);
        return M;
    } else if (model_type == 4) {
        cv::Mat src(5,2,CV_32FC1, standard_face_ffhq);
        memcpy(src.data, standard_face_ffhq, 2 * 5 * sizeof(float));
        cv::Mat M= SimilarTransform(dst, src);
        return M;
    } else {
        std::cout << "model type error." << std::endl;
        return cv::Mat();
    }
}

bool Aligner::Impl::validLmks(float landmarks[10]) {
    // cv::Point2f left_eye = cv::Point2f(landmarks[0], landmarks[1]);
    // cv::Point2f right_eye = cv::Point2f(landmarks[2], landmarks[3]);
    // cv::Point2f nose_tip = cv::Point2f(landmarks[4], landmarks[5]);
    // cv::Point2f mouth_left = cv::Point2f(landmarks[6], landmarks[7]);
    // cv::Point2f mouth_right = cv::Point2f(landmarks[8], landmarks[9]);

    // cv::Point2f eye_center = (left_eye + right_eye) / 2.0f;

    // cv::Point2f eye_to_nose = nose_tip - eye_center;
    // double yaw = atan2(eye_to_nose.y, eye_to_nose.x);
    // double yaw_degrees = yaw  / CV_PI;  // 将弧度转换为度

    // cv::Point2f mouth_center = (mouth_left + mouth_right) / 2.0f;
    // cv::Point2f nose_to_mouth = mouth_center - nose_tip;
    // double pitch = atan2(nose_to_mouth.y, nose_to_mouth.x);
    // double pitch_degrees = pitch  / CV_PI; 

    // std::cout<<"degrees: "<< yaw_degrees<<" "<<pitch_degrees<<std::endl;
    // double alpha = 10.0 / 3.0;
    // double beta = -7.0 / 3.0;
    // double final_score = 0.5 * (alpha * cos(yaw_degrees) + beta) + 0.5 * (alpha * cos(pitch_degrees) + beta);
    // std::cout<<"score: "<<final_score<<std::endl;
    float lmks[5][2];
    for (unsigned int i=0;i<10;i++) {
        lmks[i/2][i%2] = landmarks[i];
    }

    float width_up   = lmks[1][0] - lmks[0][0];
    float width_down = lmks[4][0] - lmks[3][0];
    float height_left  = lmks[3][1] - lmks[0][1];
    float height_right = lmks[4][1] - lmks[1][1];

    if (lmks[2][0]<(lmks[0][0] + width_up / 5.f) || lmks[2][0]<(lmks[3][0] + width_down / 5.f) || 
        lmks[2][0]>(lmks[1][0] - width_up / 5.f) || lmks[2][0]>(lmks[4][0] - width_down / 5.f)){
    return false;
    }
    if (lmks[2][1]>(lmks[3][1] - height_left / 5.f) || lmks[2][1]>(lmks[4][1] - height_right / 5.f) || 
        lmks[2][1]<(lmks[1][1] + height_left / 5.f) || lmks[2][1]<(lmks[0][1] + height_right / 5.f)){
    return false;
    }
    return true;
}

// GPU-accelerated version using CUDA
cv::Mat Aligner::Impl::MeanAxis0(const cv::Mat & src) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat output(1, dim, CV_32F);
    // Ensure src is continuous and float32
    // CV_Assert(src.type() == CV_32F && src.isContinuous());
    Aligner::MeanAxis0CUDA(src.ptr<float>(), output.ptr<float>(), num, dim);
    return output;
}

// GPU-accelerated version using CUDA
cv::Mat Aligner::Impl::ElementwiseMinus(const cv::Mat & A, const cv::Mat & B) {
    // CV_Assert(A.type() == CV_32F && B.type() == CV_32F && A.isContinuous() && B.isContinuous());
    // CV_Assert(B.rows == 1 && B.cols == A.cols);
    cv::Mat output(A.rows, A.cols, CV_32F);
    Aligner::ElementwiseMinusCUDA(A.ptr<float>(), B.ptr<float>(), output.ptr<float>(), A.rows, A.cols);
    return output;
}

// GPU-accelerated version using CUDA
cv::Mat Aligner::Impl::VarAxis0(const cv::Mat & src) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat mean = MeanAxis0(src);
    cv::Mat var(1, dim, CV_32F);
    Aligner::VarAxis0CUDA(src.ptr<float>(), mean.ptr<float>(), var.ptr<float>(), num, dim);
    return var;
}

// GPU-accelerated version using cuSOLVER
int Aligner::Impl::MatrixRank(cv::Mat M) {
    // CV_Assert(M.type() == CV_32F && M.isContinuous());
    return Aligner::MatrixRankSVD_CUDA(M.ptr<float>(), M.rows, M.cols, 1e-4f);
}

/*
References: "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
Anthor: Jack Yu
*/

// Refactored: Use GPU-accelerated primitives for all major ops. Remaining CPU ops are marked for future CUDA migration.
cv::Mat Aligner::Impl::SimilarTransform(const cv::Mat & src, const cv::Mat & dst) {
    const int num = src.rows;
    const int dim = src.cols;
    // CV_Assert(src.type() == CV_32F && dst.type() == CV_32F);

    // Means (GPU)
    cv::Mat src_mean = MeanAxis0(src);
    cv::Mat dst_mean = MeanAxis0(dst);

    // Demean (GPU elementwise)
    cv::Mat src_demean = ElementwiseMinus(src, src_mean);
    cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);

    // Cross-covariance (CPU matmul; small but correct for any num)
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);

    std::vector<float> Uv, Sv, VTv;
    ::svd_full_cuda(A.ptr<float>(), A.rows, A.cols, Uv, Sv, VTv);
    cv::Mat U(A.rows, A.rows, CV_32F, Uv.data());
    cv::Mat VT(A.cols, A.cols, CV_32F, VTv.data());
    cv::Mat S((int)Sv.size(), 1, CV_32F, Sv.data());

    // Reflection correction matrix D
    cv::Mat D = cv::Mat::eye(dim, dim, CV_32F);
    if (cv::determinant(U * VT) < 0) {
        D.at<float>(dim - 1, dim - 1) = -1.0f;
    }
    // Rotation
    cv::Mat R = U * D * VT;

    // Scale using variance (GPU var + CPU sum); trace(S*D)
    cv::Mat var_ = VarAxis0(src_demean);
    float var_sum = static_cast<float>(cv::sum(var_)[0]);
    float trace_SD = 0.0f;
    for (int i = 0; i < S.rows; ++i) trace_SD += S.at<float>(i, 0) * D.at<float>(i, i);
    float scale = trace_SD / var_sum;

    // Translation
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat R_scaled = scale * R;
    R_scaled.copyTo(T.rowRange(0, dim).colRange(0, dim));
    cv::Mat t = dst_mean.t() - R_scaled * src_mean.t();
    t.copyTo(T.rowRange(0, dim).colRange(dim, dim + 1));
    return T;
}

}


