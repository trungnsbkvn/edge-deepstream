/*
 * @Author: zhouyuchong
 * @Date: 2024-02-26 14:51:58
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-08-16 11:38:39
 */
#ifndef _DAMONZZZ_ALIGNER_H_
#define _DAMONZZZ_ALIGNER_H_

#include "opencv2/opencv.hpp"
#include <cuda_runtime_api.h> // for cudaStream_t

namespace alignnamespace {
class Aligner {
public:
	Aligner();
	~Aligner();

	cv::Mat Align(const cv::Mat & dst, int model_type);
	bool validLmks(float landmarks[10]);

	// GPU-accelerated primitives
	static void MeanAxis0CUDA(const float* h_src, float* h_mean, int num_rows, int num_cols, cudaStream_t stream = 0);
	static void ElementwiseMinusCUDA(const float* h_A, const float* h_B, float* h_output, int rows, int cols, cudaStream_t stream = 0);
	static void VarAxis0CUDA(const float* h_src, const float* h_mean, float* h_var, int num_rows, int num_cols, cudaStream_t stream = 0);
	static int MatrixRankSVD_CUDA(const float* h_A, int rows, int cols, float tol = 1e-4);

	// CUDA matrix ops for 2x2 and 3x3 (row-major)
	static void MatMul2x2CUDA(const float* h_A, const float* h_B, float* h_C);
	static void Det2x2CUDA(const float* h_A, float* h_det);
	static void Transpose2x2CUDA(const float* h_A, float* h_At);

	static void MatMul3x3CUDA(const float* h_A, const float* h_B, float* h_C);
	static void Det3x3CUDA(const float* h_A, float* h_det);
	static void Transpose3x3CUDA(const float* h_A, float* h_At);

	// New: GPU warpAffine for interleaved uint8 3/4 channel surfaces (src/dst on device)
	// invM2x3 is host pointer to inverse 2x3 matrix (dst->src)
	static void WarpAffineU8CUDA(const uint8_t* d_src, int src_w, int src_h, int src_pitch,
	                             uint8_t* d_dst, int dst_w, int dst_h, int dst_pitch,
	                             const float* invM2x3, int channels, uint8_t border_value = 0, cudaStream_t stream = 0);

private:
	class Impl;
	Impl* impl_;
};

} // namespace alignnamespace

#endif // !_DAMONZZZ_ALIGNER_H_