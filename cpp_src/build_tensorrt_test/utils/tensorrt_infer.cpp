#include "tensorrt_infer.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

// TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return {}; \
        } \
    } while(0)

#endif

namespace EdgeDeepStream {

TensorRTInfer::TensorRTInfer() 
    : runtime_(nullptr), engine_(nullptr), context_(nullptr), logger_(nullptr)
    , batch_size_(0), input_size_(0), output_size_(0), initialized_(false) {
}

TensorRTInfer::~TensorRTInfer() {
    shutdown();
}

bool TensorRTInfer::initialize(const std::string& engine_path, const std::string& /* mode */) {
#ifdef HAVE_TENSORRT
    try {
        engine_path_ = engine_path;
        
        // Create logger
        logger_ = new Logger();
        auto* logger = static_cast<Logger*>(logger_);
        
        // Create runtime
        runtime_ = nvinfer1::createInferRuntime(*logger);
        if (!runtime_) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        auto* runtime = static_cast<nvinfer1::IRuntime*>(runtime_);
        
        // Load engine from file
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            std::cerr << "Failed to open engine file: " << engine_path << std::endl;
            return false;
        }
        
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(engine_size);
        engine_file.read(engine_data.data(), engine_size);
        engine_file.close();
        
        // Deserialize engine
        engine_ = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
        if (!engine_) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }
        auto* engine = static_cast<nvinfer1::ICudaEngine*>(engine_);
        
        // Create execution context
        context_ = engine->createExecutionContext();
        if (!context_) {
            std::cerr << "Failed to create TensorRT execution context" << std::endl;
            return false;
        }
        
        // Setup bindings and allocate buffers
        if (!setup_bindings()) {
            std::cerr << "Failed to setup TensorRT bindings" << std::endl;
            return false;
        }
        
        if (!allocate_buffers()) {
            std::cerr << "Failed to allocate TensorRT buffers" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "TensorRT engine initialized successfully: " << engine_path << std::endl;
        std::cout << "  Batch size: " << batch_size_ << std::endl;
        std::cout << "  Input size: " << input_size_ << std::endl;
        std::cout << "  Output size: " << output_size_ << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "TensorRT initialization error: " << e.what() << std::endl;
        return false;
    }
#else
    // Placeholder implementation when TensorRT not available
    engine_path_ = engine_path;
    initialized_ = true;
    batch_size_ = 1;
    input_size_ = 3 * 112 * 112;  // ArcFace input: 3x112x112
    output_size_ = 512;           // ArcFace output: 512-dim features
    
    std::cout << "TensorRT not available - using placeholder implementation" << std::endl;
    std::cout << "Engine path (placeholder): " << engine_path << std::endl;
    return true;
#endif
}

bool TensorRTInfer::setup_bindings() {
#ifdef HAVE_TENSORRT
    auto* engine = static_cast<nvinfer1::ICudaEngine*>(engine_);
    auto* context = static_cast<nvinfer1::IExecutionContext*>(context_);
    
    int num_bindings = engine->getNbIOTensors();
    bindings_.clear();
    bindings_.reserve(num_bindings);
    
    for (int i = 0; i < num_bindings; ++i) {
        const char* tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        bool is_input = (io_mode == nvinfer1::TensorIOMode::kINPUT);
        
        nvinfer1::Dims dims = context->getTensorShape(tensor_name);
        
        // Handle dynamic shapes by setting batch size to 1
        if (is_input && dims.d[0] == -1) {
            nvinfer1::Dims input_dims = dims;
            input_dims.d[0] = 1;  // Set batch size to 1
            context->setInputShape(tensor_name, input_dims);
            dims = context->getTensorShape(tensor_name);  // Get updated dims
        }
        
        // nvinfer1::DataType dtype = engine->getTensorDataType(tensor_name);
        
        // Calculate size (handle dynamic batch dimension)
        size_t size = 1;
        std::vector<int> shape;
        for (int j = 0; j < dims.nbDims; ++j) {
            int dim = dims.d[j];
            // Handle dynamic dimension (-1) by setting to batch size 1
            if (dim == -1) {
                dim = 1;
            }
            shape.push_back(dim);
            size *= dim;
        }
        
        if (is_input) {
            batch_size_ = dims.d[0];
            input_size_ = size;
        } else {
            output_size_ = size;
        }
        
        size_t nbytes = size * sizeof(float);  // Assume float32
        
        Binding binding;
        binding.size = size;
        binding.nbytes = nbytes;
        binding.dims = shape;
        binding.name = tensor_name;
        binding.is_input = is_input;
        binding.device_ptr = nullptr;
        binding.host_ptr = nullptr;
        
        bindings_.push_back(binding);
        
        std::cout << "Binding " << i << " (" << (is_input ? "Input" : "Output") << "): "
                  << tensor_name << " [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << "x";
        }
        std::cout << "] = " << size << " elements" << std::endl;
    }
    
    return true;
#else
    return true;
#endif
}

bool TensorRTInfer::allocate_buffers() {
#ifdef HAVE_TENSORRT
    device_ptrs_.clear();
    
    for (auto& binding : bindings_) {
        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&binding.device_ptr, binding.nbytes));
        device_ptrs_.push_back(binding.device_ptr);
        
        // Allocate host memory for outputs
        if (!binding.is_input) {
            binding.host_ptr = malloc(binding.nbytes);
            if (!binding.host_ptr) {
                std::cerr << "Failed to allocate host memory for " << binding.name << std::endl;
                return false;
            }
        }
    }
    
    return true;
#else
    return true;
#endif
}

void TensorRTInfer::cleanup_buffers() {
#ifdef HAVE_TENSORRT
    for (auto& binding : bindings_) {
        if (binding.device_ptr) {
            cudaFree(binding.device_ptr);
            binding.device_ptr = nullptr;
        }
        if (binding.host_ptr) {
            free(binding.host_ptr);
            binding.host_ptr = nullptr;
        }
    }
    device_ptrs_.clear();
    bindings_.clear();
#endif
}

std::vector<float> TensorRTInfer::infer(const std::vector<float>& input) {
#ifdef HAVE_TENSORRT
    if (!initialized_ || bindings_.empty()) {
        std::cerr << "TensorRT engine not initialized" << std::endl;
        return {};
    }
    
    // Find input and output bindings
    Binding* input_binding = nullptr;
    Binding* output_binding = nullptr;
    
    for (auto& binding : bindings_) {
        if (binding.is_input) {
            input_binding = &binding;
        } else {
            output_binding = &binding;
        }
    }
    
    if (!input_binding || !output_binding) {
        std::cerr << "Failed to find input/output bindings" << std::endl;
        return {};
    }
    
    // Validate input size
    if (input.size() != input_binding->size) {
        std::cerr << "Input size mismatch: expected " << input_binding->size 
                  << ", got " << input.size() << std::endl;
        return {};
    }
    
    try {
        // Copy input to device
        CHECK_CUDA(cudaMemcpy(input_binding->device_ptr, input.data(), 
                   input_binding->nbytes, cudaMemcpyHostToDevice));
        
        // Execute inference
        auto* context = static_cast<nvinfer1::IExecutionContext*>(context_);
        bool success = context->executeV2(device_ptrs_.data());
        if (!success) {
            std::cerr << "TensorRT inference execution failed" << std::endl;
            return {};
        }
        
        // Copy output from device to host
        CHECK_CUDA(cudaMemcpy(output_binding->host_ptr, output_binding->device_ptr, 
                   output_binding->nbytes, cudaMemcpyDeviceToHost));
        
        // Convert output to vector
        float* output_data = static_cast<float*>(output_binding->host_ptr);
        std::vector<float> result(output_data, output_data + output_binding->size);
        
        return result;
    } catch (const std::exception& e) {
        std::cerr << "TensorRT inference error: " << e.what() << std::endl;
        return {};
    }
#else
    // Placeholder implementation
    if (!initialized_) {
        std::cerr << "TensorRT engine not initialized (placeholder)" << std::endl;
        return {};
    }
    
    if (input.size() != static_cast<size_t>(input_size_)) {
        std::cerr << "Input size mismatch (placeholder): expected " << input_size_ 
                  << ", got " << input.size() << std::endl;
        return {};
    }
    
    // Generate a fake but reasonable feature vector
    std::vector<float> result(output_size_);
    
    // Create a pseudo-random but deterministic feature based on input
    float sum = 0.0f;
    for (size_t i = 0; i < input.size() && i < 100; ++i) {
        sum += input[i];
    }
    
    // Generate normalized features
    for (int i = 0; i < output_size_; ++i) {
        result[i] = std::sin(sum * 0.1f + i * 0.01f) * 0.1f;
    }
    
    // Normalize to unit length (for cosine similarity)
    float norm = 0.0f;
    for (float v : result) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-12f) {
        for (float& v : result) {
            v /= norm;
        }
    }
    
    return result;
#endif
}

std::vector<float> TensorRTInfer::preprocess_face(const cv::Mat& face_roi) {
    try {
        // Resize to 112x112 (ArcFace standard input size)
        cv::Mat resized;
        cv::resize(face_roi, resized, cv::Size(112, 112));
        
        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        
        // Normalize: (pixel - 127.5) / 128.0
        cv::Mat normalized;
        rgb.convertTo(normalized, CV_32F);
        normalized = (normalized - 127.5) / 128.0;
        
        // Convert to CHW format (Channel-Height-Width)
        std::vector<cv::Mat> channels(3);
        cv::split(normalized, channels);
        
        std::vector<float> input_data;
        input_data.reserve(3 * 112 * 112);
        
        // Add channels in CHW order
        for (int c = 0; c < 3; ++c) {
            cv::Mat channel = channels[c];
            float* data = (float*)channel.data;
            input_data.insert(input_data.end(), data, data + 112 * 112);
        }
        
        return input_data;
    } catch (const std::exception& e) {
        std::cerr << "Face preprocessing error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<float> TensorRTInfer::extract_features(const cv::Mat& face_image) {
    if (!initialized_) {
        std::cerr << "TensorRT engine not initialized" << std::endl;
        return {};
    }
    
    // Preprocess face image
    auto input_data = preprocess_face(face_image);
    if (input_data.empty()) {
        std::cerr << "Failed to preprocess face image" << std::endl;
        return {};
    }
    
    // Run inference
    return infer(input_data);
}

std::vector<int> TensorRTInfer::get_input_shape() const {
    if (!bindings_.empty()) {
        for (const auto& binding : bindings_) {
            if (binding.is_input) {
                return binding.dims;
            }
        }
    }
    return {1, 3, 112, 112};  // Default ArcFace input shape
}

std::vector<int> TensorRTInfer::get_output_shape() const {
    if (!bindings_.empty()) {
        for (const auto& binding : bindings_) {
            if (!binding.is_input) {
                return binding.dims;
            }
        }
    }
    return {1, 512};  // Default ArcFace output shape
}

void TensorRTInfer::shutdown() {
    if (!initialized_) return;
    
#ifdef HAVE_TENSORRT
    cleanup_buffers();
    
    if (context_) {
        delete static_cast<nvinfer1::IExecutionContext*>(context_);
        context_ = nullptr;
    }
    
    if (engine_) {
        delete static_cast<nvinfer1::ICudaEngine*>(engine_);
        engine_ = nullptr;
    }
    
    if (runtime_) {
        delete static_cast<nvinfer1::IRuntime*>(runtime_);
        runtime_ = nullptr;
    }
    
    if (logger_) {
        delete static_cast<Logger*>(logger_);
        logger_ = nullptr;
    }
#endif
    
    initialized_ = false;
    std::cout << "TensorRT engine shutdown complete" << std::endl;
}

// ============================================================================
// ArcFace Engine Implementation
// ============================================================================

ArcFaceEngine::ArcFaceEngine() : TensorRTInfer() {
}

bool ArcFaceEngine::initialize_arcface(const std::string& engine_path) {
    return initialize(engine_path, "FP16");
}

std::vector<float> ArcFaceEngine::preprocess_aligned_face(const cv::Mat& aligned_face) {
    try {
        // ArcFace expects 112x112 RGB input
        cv::Mat resized;
        if (aligned_face.size() != cv::Size(112, 112)) {
            cv::resize(aligned_face, resized, cv::Size(112, 112));
        } else {
            resized = aligned_face.clone();
        }
        
        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        
        // Normalize: (pixel - 127.5) / 128.0 (ArcFace normalization)
        cv::Mat normalized;
        rgb.convertTo(normalized, CV_32F);
        normalized = (normalized - 127.5) / 128.0;
        
        // Convert to CHW format (Channel-Height-Width)
        std::vector<cv::Mat> channels(3);
        cv::split(normalized, channels);
        
        std::vector<float> input_data;
        input_data.reserve(3 * 112 * 112);
        
        // Add channels in CHW order (R, G, B)
        for (int c = 0; c < 3; ++c) {
            cv::Mat channel = channels[c];
            float* data = (float*)channel.data;
            input_data.insert(input_data.end(), data, data + 112 * 112);
        }
        
        return input_data;
    } catch (const std::exception& e) {
        std::cerr << "ArcFace preprocessing error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<float> ArcFaceEngine::extract_face_features(const cv::Mat& aligned_face) {
    if (!initialized_) {
        std::cerr << "ArcFace engine not initialized" << std::endl;
        return {};
    }
    
    // Preprocess aligned face
    auto input_data = preprocess_aligned_face(aligned_face);
    if (input_data.empty()) {
        std::cerr << "Failed to preprocess aligned face for ArcFace" << std::endl;
        return {};
    }
    
    // Run inference
    auto features = infer(input_data);
    
    // ArcFace features should be normalized (unit vector)
    if (!features.empty() && features.size() == 512) {
        float norm = 0.0f;
        for (float f : features) {
            norm += f * f;
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-12f) {
            for (float& f : features) {
                f /= norm;
            }
        }
    }
    
    return features;
}

// ============================================================================
// YOLOv8n Face Detection Engine Implementation
// ============================================================================

YOLOv8FaceEngine::YOLOv8FaceEngine() : TensorRTInfer() {
}

bool YOLOv8FaceEngine::initialize_yolo_face(const std::string& engine_path) {
    return initialize(engine_path, "FP16");
}

std::vector<float> YOLOv8FaceEngine::preprocess_detection_image(const cv::Mat& image, int input_width, int input_height) {
    try {
        // Resize with padding to maintain aspect ratio
        cv::Mat resized;
        float scale = std::min(input_width / (float)image.cols, input_height / (float)image.rows);
        int new_width = (int)(image.cols * scale);
        int new_height = (int)(image.rows * scale);
        
        cv::resize(image, resized, cv::Size(new_width, new_height));
        
        // Create padded image
        cv::Mat padded = cv::Mat::zeros(input_height, input_width, CV_8UC3);
        int x_offset = (input_width - new_width) / 2;
        int y_offset = (input_height - new_height) / 2;
        
        resized.copyTo(padded(cv::Rect(x_offset, y_offset, new_width, new_height)));
        
        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
        
        // Normalize: pixel / 255.0 (YOLOv8 normalization)
        cv::Mat normalized;
        rgb.convertTo(normalized, CV_32F, 1.0/255.0);
        
        // Convert to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(normalized, channels);
        
        std::vector<float> input_data;
        input_data.reserve(3 * input_width * input_height);
        
        // Add channels in CHW order
        for (int c = 0; c < 3; ++c) {
            cv::Mat channel = channels[c];
            float* data = (float*)channel.data;
            input_data.insert(input_data.end(), data, data + input_width * input_height);
        }
        
        return input_data;
    } catch (const std::exception& e) {
        std::cerr << "YOLOv8 preprocessing error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<int> YOLOv8FaceEngine::apply_nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float nms_threshold) {
    std::vector<int> indices;
    std::vector<std::pair<float, int>> score_index_pairs;
    
    // Create score-index pairs and sort by score (descending)
    for (int i = 0; i < (int)scores.size(); ++i) {
        score_index_pairs.push_back({scores[i], i});
    }
    std::sort(score_index_pairs.rbegin(), score_index_pairs.rend());
    
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (const auto& pair : score_index_pairs) {
        int idx = pair.second;
        if (suppressed[idx]) continue;
        
        indices.push_back(idx);
        
        // Suppress overlapping boxes
        for (int i = 0; i < (int)boxes.size(); ++i) {
            if (i == idx || suppressed[i]) continue;
            
            // Calculate IoU
            cv::Rect intersection = boxes[idx] & boxes[i];
            float intersection_area = intersection.area();
            float union_area = boxes[idx].area() + boxes[i].area() - intersection_area;
            
            if (union_area > 0) {
                float iou = intersection_area / union_area;
                if (iou > nms_threshold) {
                    suppressed[i] = true;
                }
            }
        }
    }
    
    return indices;
}

std::vector<FaceDetection> YOLOv8FaceEngine::postprocess_detections(
    const std::vector<float>& boxes,
    const std::vector<float>& scores, 
    const std::vector<float>& landmarks,
    int input_width, int input_height,
    int original_width, int original_height,
    float conf_threshold,
    float nms_threshold) {
    
    std::vector<FaceDetection> detections;
    std::vector<cv::Rect> valid_boxes;
    std::vector<float> valid_scores;
    std::vector<std::vector<cv::Point2f>> valid_landmarks;
    
    // Calculate scale factors
    float scale_x = (float)original_width / input_width;
    float scale_y = (float)original_height / input_height;
    
    // Filter detections by confidence threshold
    size_t num_detections = scores.size();
    for (size_t i = 0; i < num_detections; ++i) {
        if (scores[i] > conf_threshold) {
            // Extract bounding box (assuming boxes format: [x1, y1, x2, y2])
            int x1 = (int)(boxes[i * 4 + 0] * scale_x);
            int y1 = (int)(boxes[i * 4 + 1] * scale_y);
            int x2 = (int)(boxes[i * 4 + 2] * scale_x);
            int y2 = (int)(boxes[i * 4 + 3] * scale_y);
            
            // Clamp to image boundaries
            x1 = std::max(0, std::min(x1, original_width));
            y1 = std::max(0, std::min(y1, original_height));
            x2 = std::max(0, std::min(x2, original_width));
            y2 = std::max(0, std::min(y2, original_height));
            
            if (x2 > x1 && y2 > y1) {
                cv::Rect bbox(x1, y1, x2 - x1, y2 - y1);
                valid_boxes.push_back(bbox);
                valid_scores.push_back(scores[i]);
                
                // Extract landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
                std::vector<cv::Point2f> face_landmarks;
                for (int j = 0; j < 5; ++j) {
                    float lx = landmarks[i * 10 + j * 2 + 0] * scale_x;
                    float ly = landmarks[i * 10 + j * 2 + 1] * scale_y;
                    face_landmarks.push_back(cv::Point2f(lx, ly));
                }
                valid_landmarks.push_back(face_landmarks);
            }
        }
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> nms_indices = apply_nms(valid_boxes, valid_scores, nms_threshold);
    
    // Create final detections
    for (int idx : nms_indices) {
        FaceDetection detection;
        detection.bbox = valid_boxes[idx];
        detection.confidence = valid_scores[idx];
        detection.landmarks = valid_landmarks[idx];
        detections.push_back(detection);
    }
    
    return detections;
}

std::vector<FaceDetection> YOLOv8FaceEngine::detect_faces(const cv::Mat& image, float conf_threshold, float nms_threshold) {
    if (!initialized_) {
        std::cerr << "YOLOv8 Face engine not initialized" << std::endl;
        return {};
    }
    
    // Get input dimensions (typically 640x640 for YOLOv8n)
    auto input_shape = get_input_shape();
    int input_width = 640, input_height = 640;
    if (input_shape.size() >= 4) {
        input_height = input_shape[2];
        input_width = input_shape[3];
    }
    
    // Preprocess image
    auto input_data = preprocess_detection_image(image, input_width, input_height);
    if (input_data.empty()) {
        std::cerr << "Failed to preprocess image for YOLOv8 detection" << std::endl;
        return {};
    }
    
    // Run inference
    auto outputs = infer(input_data);
    if (outputs.empty()) {
        std::cerr << "YOLOv8 inference failed" << std::endl;
        return {};
    }
    
    // Parse outputs (YOLOv8 typically outputs: boxes, scores, landmarks)
    // This is a simplified parsing - actual format depends on your specific YOLOv8 model
    size_t num_detections = outputs.size() / 15;  // Assuming 4 box coords + 1 score + 10 landmarks = 15 values per detection
    
    std::vector<float> boxes, scores, landmarks;
    for (size_t i = 0; i < num_detections; ++i) {
        size_t base_idx = i * 15;
        
        // Extract box coordinates (x1, y1, x2, y2)
        boxes.insert(boxes.end(), outputs.begin() + base_idx, outputs.begin() + base_idx + 4);
        
        // Extract score
        scores.push_back(outputs[base_idx + 4]);
        
        // Extract landmarks (10 values: 5 points x 2 coordinates)
        landmarks.insert(landmarks.end(), outputs.begin() + base_idx + 5, outputs.begin() + base_idx + 15);
    }
    
    return postprocess_detections(boxes, scores, landmarks, 
                                input_width, input_height,
                                image.cols, image.rows,
                                conf_threshold, nms_threshold);
}

} // namespace EdgeDeepStream