#include "../align_functions.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using alignnamespace::Aligner;

static cv::Mat cpu_similar_transform(const cv::Mat& src, const cv::Mat& dst) {
    const int num = src.rows;
    const int dim = src.cols;
    cv::Mat src_mean, dst_mean;
    cv::reduce(src, src_mean, 0, cv::REDUCE_AVG);
    cv::reduce(dst, dst_mean, 0, cv::REDUCE_AVG);
    cv::Mat src_demean = src - cv::repeat(src_mean, src.rows, 1);
    cv::Mat dst_demean = dst - cv::repeat(dst_mean, dst.rows, 1);

    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat U, S, VT;
    cv::SVD::compute(A, S, U, VT);
    cv::Mat D = cv::Mat::eye(dim, dim, CV_32F);
    if (cv::determinant(U * VT) < 0) D.at<float>(dim - 1, dim - 1) = -1.0f;
    cv::Mat R = U * D * VT;
    cv::Mat var_;
    cv::reduce(src_demean.mul(src_demean), var_, 0, cv::REDUCE_AVG);
    float var_sum = static_cast<float>(cv::sum(var_)[0]);
    float trace_SD = 0.0f; for (int i = 0; i < S.rows; ++i) trace_SD += S.at<float>(i, 0) * D.at<float>(i, i);
    float scale = trace_SD / var_sum;
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat R_scaled = scale * R;
    R_scaled.copyTo(T.rowRange(0, dim).colRange(0, dim));
    cv::Mat t = dst_mean.t() - R_scaled * src_mean.t();
    t.copyTo(T.rowRange(0, dim).colRange(dim, dim + 1));
    return T;
}

int main() {
    // Use a deterministic 5-point test case (face landmarks)
    float src_pts_data[5][2] = {
        {30.1f, 52.3f}, {72.2f, 50.9f}, {55.0f, 70.7f}, {40.0f, 92.0f}, {69.5f, 91.3f}
    };
    float dst_pts_data[5][2] = {
        {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
    };
    cv::Mat src(5, 2, CV_32F, src_pts_data);
    cv::Mat dst(5, 2, CV_32F, dst_pts_data);

    // GPU path via Aligner public API (face model_type=1)
    Aligner aligner;
    cv::Mat T_gpu = aligner.Align(src, 1);

    // CPU reference
    cv::Mat T_cpu = cpu_similar_transform(src, dst);

    double diff_norm = cv::norm(T_cpu - T_gpu, cv::NORM_INF);
    std::cout << "T_cpu:\n" << T_cpu << "\n\nT_gpu:\n" << T_gpu << "\n\nmax_abs_diff=" << diff_norm << std::endl;

    // Simple threshold check
    const double tol = 1e-3; // small differences are expected due to GPU/CPU math
    if (diff_norm < tol) {
        std::cout << "RESULT: PASS" << std::endl;
        return 0;
    } else {
        std::cout << "RESULT: FAIL" << std::endl;
        return 1;
    }
}
