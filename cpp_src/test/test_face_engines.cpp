#include "tensorrt_infer.h"
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace EdgeDeepStream;

void test_arcface_engine() {
    std::cout << "\n=== Testing ArcFace Engine ===" << std::endl;
    
    ArcFaceEngine arcface;
    
    // Use the batch size 1 engine (more manageable for testing)
    std::string arcface_engine = "/home/m2n/edge-deepstream/models/arcface/arcface.engine";
    
    std::cout << "Initializing ArcFace engine: " << arcface_engine << std::endl;
    bool success = arcface.initialize_arcface(arcface_engine);
    std::cout << "ArcFace initialization: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    
    if (success && arcface.is_initialized()) {
        auto input_shape = arcface.get_input_shape();
        auto output_shape = arcface.get_output_shape();
        
        std::cout << "ArcFace Input shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "ArcFace Output shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Test with a sample face image (112x112)
        cv::Mat test_face = cv::Mat::zeros(112, 112, CV_8UC3);
        // Add some pattern
        cv::randu(test_face, cv::Scalar::all(50), cv::Scalar::all(200));
        
        std::cout << "Extracting face features..." << std::endl;
        auto features = arcface.extract_face_features(test_face);
        
        if (!features.empty()) {
            std::cout << "Feature extraction: SUCCESS" << std::endl;
            std::cout << "Feature dimension: " << features.size() << std::endl;
            
            // Calculate L2 norm (should be ~1.0 for normalized features)
            float norm = 0.0f;
            for (float f : features) {
                norm += f * f;
            }
            norm = std::sqrt(norm);
            std::cout << "Feature vector L2 norm: " << std::fixed << std::setprecision(6) << norm << std::endl;
            
            // Show first few features
            std::cout << "First 5 features: [";
            for (int i = 0; i < std::min(5, (int)features.size()); ++i) {
                std::cout << std::setprecision(4) << features[i];
                if (i < 4) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "Feature extraction: FAILED" << std::endl;
        }
        
        arcface.shutdown();
    } else {
        std::cout << "Skipping ArcFace tests - initialization failed" << std::endl;
    }
}

void test_yolov8_engine() {
    std::cout << "\n=== Testing YOLOv8n Face Detection Engine ===" << std::endl;
    
    YOLOv8FaceEngine yolo;
    
    // Use the batch size 1 engine (more manageable for testing)
    std::string yolo_engine = "/home/m2n/edge-deepstream/models/yolov8n_face/yolov8n-face.onnx_b1_gpu0_fp16.engine";
    
    std::cout << "Initializing YOLOv8n engine: " << yolo_engine << std::endl;
    bool success = yolo.initialize_yolo_face(yolo_engine);
    std::cout << "YOLOv8n initialization: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    
    if (success && yolo.is_initialized()) {
        auto input_shape = yolo.get_input_shape();
        auto output_shape = yolo.get_output_shape();
        
        std::cout << "YOLOv8n Input shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "YOLOv8n Output shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Test with a sample image
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        // Add some patterns that might look like faces
        cv::rectangle(test_image, cv::Rect(100, 100, 80, 100), cv::Scalar(180, 150, 120), -1);
        cv::rectangle(test_image, cv::Rect(400, 200, 70, 90), cv::Scalar(200, 170, 140), -1);
        
        std::cout << "Detecting faces in test image..." << std::endl;
        auto detections = yolo.detect_faces(test_image, 0.35f, 0.45f);
        
        std::cout << "Face detection: " << (detections.empty() ? "NO FACES FOUND" : "SUCCESS") << std::endl;
        std::cout << "Number of faces detected: " << detections.size() << std::endl;
        
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            std::cout << "Face " << i+1 << ":" << std::endl;
            std::cout << "  Bbox: (" << det.bbox.x << ", " << det.bbox.y 
                      << ", " << det.bbox.width << ", " << det.bbox.height << ")" << std::endl;
            std::cout << "  Confidence: " << std::setprecision(3) << det.confidence << std::endl;
            std::cout << "  Landmarks: ";
            for (size_t j = 0; j < det.landmarks.size(); ++j) {
                std::cout << "(" << (int)det.landmarks[j].x << ", " << (int)det.landmarks[j].y << ")";
                if (j < det.landmarks.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        yolo.shutdown();
    } else {
        std::cout << "Skipping YOLOv8n tests - initialization failed" << std::endl;
    }
}

void test_combined_pipeline() {
    std::cout << "\n=== Testing Combined Face Recognition Pipeline ===" << std::endl;
    
    // Initialize both engines
    YOLOv8FaceEngine detector;
    ArcFaceEngine feature_extractor;
    
    std::string yolo_engine = "/home/m2n/edge-deepstream/models/yolov8n_face/yolov8n-face.onnx_b1_gpu0_fp16.engine";
    std::string arcface_engine = "/home/m2n/edge-deepstream/models/arcface/arcface.engine";
    
    bool yolo_ok = detector.initialize_yolo_face(yolo_engine);
    bool arcface_ok = feature_extractor.initialize_arcface(arcface_engine);
    
    if (yolo_ok && arcface_ok) {
        std::cout << "Both engines initialized successfully!" << std::endl;
        
        // Create a test image with simulated face
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::rectangle(test_image, cv::Rect(200, 150, 120, 150), cv::Scalar(180, 150, 120), -1);
        
        // Step 1: Detect faces
        auto faces = detector.detect_faces(test_image);
        std::cout << "Detected " << faces.size() << " face(s)" << std::endl;
        
        // Step 2: Extract features for each detected face
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            
            // Extract face ROI and resize to 112x112 for ArcFace
            cv::Mat face_roi = test_image(face.bbox);
            cv::Mat aligned_face;
            cv::resize(face_roi, aligned_face, cv::Size(112, 112));
            
            // Extract features
            auto features = feature_extractor.extract_face_features(aligned_face);
            
            if (!features.empty()) {
                std::cout << "Face " << i+1 << " features extracted: " << features.size() << " dimensions" << std::endl;
                
                float norm = 0.0f;
                for (float f : features) {
                    norm += f * f;
                }
                std::cout << "  Feature vector norm: " << std::sqrt(norm) << std::endl;
            }
        }
        
        detector.shutdown();
        feature_extractor.shutdown();
        std::cout << "Combined pipeline test complete!" << std::endl;
    } else {
        std::cout << "Failed to initialize engines: YOLOv8=" << yolo_ok << ", ArcFace=" << arcface_ok << std::endl;
    }
}

int main() {
    std::cout << "=== TensorRT Face Recognition Engine Test ===" << std::endl;
    
    // Test individual engines
    test_arcface_engine();
    test_yolov8_engine();
    
    // Test combined pipeline
    test_combined_pipeline();
    
    std::cout << "\n=== All Tests Complete ===" << std::endl;
    return 0;
}