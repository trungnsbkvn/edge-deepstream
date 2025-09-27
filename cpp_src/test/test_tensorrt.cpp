#include "tensorrt_infer.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace EdgeDeepStream;

int main(int /* argc */, char* /* argv */[]) {
    std::cout << "=== TensorRT Inference Test ===" << std::endl;
    
    // Create TensorRT inference engine
    TensorRTInfer tensorrt;
    
    // Test engine paths (you would use your actual engine files)
    std::string engine_path = "../models/arcface/arcface.engine";
    
    std::cout << "Testing TensorRT initialization..." << std::endl;
    bool init_success = tensorrt.initialize(engine_path);
    
    if (!init_success) {
        std::cout << "Note: TensorRT engine not found at " << engine_path << std::endl;
        std::cout << "This is expected - testing with placeholder implementation" << std::endl;
    }
    
    std::cout << "Engine initialized: " << (tensorrt.is_initialized() ? "YES" : "NO") << std::endl;
    
    if (tensorrt.is_initialized()) {
        // Test input/output shapes
        auto input_shape = tensorrt.get_input_shape();
        auto output_shape = tensorrt.get_output_shape();
        
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Create a test face image (112x112 RGB)
        cv::Mat test_face = cv::Mat::zeros(112, 112, CV_8UC3);
        
        // Add some pattern to the test image
        for (int y = 0; y < 112; ++y) {
            for (int x = 0; x < 112; ++x) {
                test_face.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    (x + y) % 255,
                    (x * 2) % 255,
                    (y * 2) % 255
                );
            }
        }
        
        std::cout << "Running face preprocessing..." << std::endl;
        auto preprocessed = tensorrt.preprocess_face(test_face);
        std::cout << "Preprocessed data size: " << preprocessed.size() << std::endl;
        
        if (!preprocessed.empty()) {
            std::cout << "Running inference..." << std::endl;
            auto features = tensorrt.infer(preprocessed);
            
            if (!features.empty()) {
                std::cout << "Feature extraction successful!" << std::endl;
                std::cout << "Feature vector size: " << features.size() << std::endl;
                
                // Print first few and last few features
                std::cout << "First 5 features: [";
                for (int i = 0; i < std::min(5, (int)features.size()); ++i) {
                    std::cout << features[i];
                    if (i < 4 && i < (int)features.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                if (features.size() > 10) {
                    std::cout << "Last 5 features: [";
                    int start = features.size() - 5;
                    for (int i = start; i < (int)features.size(); ++i) {
                        std::cout << features[i];
                        if (i < (int)features.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
                
                // Calculate norm (should be close to 1.0 for normalized features)
                float norm = 0.0f;
                for (float f : features) {
                    norm += f * f;
                }
                norm = std::sqrt(norm);
                std::cout << "Feature vector norm: " << norm << std::endl;
                
                // Test extract_features method as well
                std::cout << "Testing extract_features method..." << std::endl;
                auto features2 = tensorrt.extract_features(test_face);
                
                if (features2.size() == features.size()) {
                    bool same = true;
                    for (size_t i = 0; i < features.size(); ++i) {
                        if (std::abs(features[i] - features2[i]) > 1e-6f) {
                            same = false;
                            break;
                        }
                    }
                    std::cout << "extract_features produces " 
                              << (same ? "identical" : "different") 
                              << " results" << std::endl;
                }
            } else {
                std::cout << "ERROR: Inference returned empty features" << std::endl;
                return 1;
            }
        } else {
            std::cout << "ERROR: Preprocessing failed" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Skipping inference tests - engine not initialized" << std::endl;
    }
    
    std::cout << "Shutting down TensorRT engine..." << std::endl;
    tensorrt.shutdown();
    
    std::cout << "=== TensorRT Test Complete ===" << std::endl;
    return 0;
}