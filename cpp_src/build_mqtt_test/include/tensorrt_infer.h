#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Forward declarations for TensorRT types when available
#ifdef HAVE_TENSORRT
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
    class ILogger;
}
#endif

namespace EdgeDeepStream {

// Structure for face detection results
struct FaceDetection {
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Point2f> landmarks;  // 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
};

/**
 * Base TensorRT inference engine for face feature extraction and detection
 * Handles model loading, preprocessing, inference, and postprocessing
 */
class TensorRTInfer {
public:
    struct Binding {
        size_t size;
        size_t nbytes;
        std::vector<int> dims;
        std::string name;
        bool is_input;
        void* device_ptr;
        void* host_ptr;
    };
    
    TensorRTInfer();
    ~TensorRTInfer();
    
    // Disable copy, enable move
    TensorRTInfer(const TensorRTInfer&) = delete;
    TensorRTInfer& operator=(const TensorRTInfer&) = delete;
    TensorRTInfer(TensorRTInfer&&) = default;
    TensorRTInfer& operator=(TensorRTInfer&&) = default;
    
    // Initialize TensorRT engine from file
    bool initialize(const std::string& engine_path, const std::string& mode = "FP32");
    
    // Run inference on preprocessed data
    std::vector<float> infer(const std::vector<float>& input);
    
    // Extract features from face image
    std::vector<float> extract_features(const cv::Mat& face_image);
    
    // Preprocess face image for inference
    std::vector<float> preprocess_face(const cv::Mat& face_roi);
    
    // Get input/output tensor shapes
    std::vector<int> get_input_shape() const;
    std::vector<int> get_output_shape() const;
    
    // Information
    bool is_initialized() const { return initialized_; }
    
    // Cleanup resources
    void shutdown();
    
private:
    bool setup_bindings();
    bool allocate_buffers();
    void cleanup_buffers();
    
protected:
    std::string engine_path_;
    void* runtime_;
    void* engine_;
    void* context_;
    void* logger_;
    
    std::vector<Binding> bindings_;
    std::vector<void*> device_ptrs_;
    
    int batch_size_;
    int input_size_;
    int output_size_;
    bool initialized_;
};

/**
 * ArcFace TensorRT engine for face feature extraction
 */
class ArcFaceEngine : public TensorRTInfer {
public:
    ArcFaceEngine();
    ~ArcFaceEngine() = default;
    
    // Initialize with ArcFace engine file
    bool initialize_arcface(const std::string& engine_path = "../models/arcface/glintr100.onnx_b4_gpu0_fp16.engine");
    
    // Extract 512-dimensional face features
    std::vector<float> extract_face_features(const cv::Mat& aligned_face);
    
    // Preprocess aligned face for ArcFace (112x112, normalized)
    std::vector<float> preprocess_aligned_face(const cv::Mat& aligned_face);
};

/**
 * YOLOv8n Face Detection TensorRT engine
 */
class YOLOv8FaceEngine : public TensorRTInfer {
public:
    YOLOv8FaceEngine();
    ~YOLOv8FaceEngine() = default;
    
    // Initialize with YOLOv8n face detection engine
    bool initialize_yolo_face(const std::string& engine_path = "../models/yolov8n_face/yolov8n-face.onnx_b4_gpu0_fp16.engine");
    
    // Detect faces in image and return bounding boxes with landmarks
    std::vector<FaceDetection> detect_faces(const cv::Mat& image, float conf_threshold = 0.35f, float nms_threshold = 0.45f);
    
    // Preprocess image for YOLOv8n detection
    std::vector<float> preprocess_detection_image(const cv::Mat& image, int input_width = 640, int input_height = 640);
    
    // Postprocess YOLOv8n outputs to face detections
    std::vector<FaceDetection> postprocess_detections(
        const std::vector<float>& boxes,
        const std::vector<float>& scores, 
        const std::vector<float>& landmarks,
        int input_width, int input_height,
        int original_width, int original_height,
        float conf_threshold = 0.35f,
        float nms_threshold = 0.45f
    );
    
private:
    // Apply Non-Maximum Suppression
    std::vector<int> apply_nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float nms_threshold);
};

} // namespace EdgeDeepStream