#include "enroll_ops.h"
#include "faiss_index.h"
#include "tensorrt_infer.h"
#include "config_parser.h"
#include "env_utils.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace EdgeDeepStream;

int main() {
    try {
        std::cout << "=== Real Face Enrollment Test with 112x112 Export ===" << std::endl;

        // Create output directory for 112x112 aligned faces
        std::string aligned_output_dir = "test_output/aligned_112x112";
        fs::create_directories(aligned_output_dir);
        std::cout << "Created output directory: " << aligned_output_dir << std::endl;

        // Initialize enrollment operations
        EnrollOps enrollOps;
        if (!enrollOps.initialize("config/config_pipeline.toml")) {
            std::cerr << "Failed to initialize enrollment operations" << std::endl;
            return 1;
        }
        std::cout << "✅ Enrollment system initialized successfully" << std::endl;

        // Initialize FAISS index
        auto faceIndex = std::make_shared<FaceIndex>(512); // 512-dimensional embeddings
        if (!faceIndex->load("data/index/faiss.index", "data/index/labels.json")) {
            std::cout << "⚠️ Could not load existing FAISS index - will create new one" << std::endl;
        }
        std::cout << "✅ FAISS index initialized successfully" << std::endl;

        // Initialize TensorRT engine for ArcFace
        auto tensorrtEngine = std::make_shared<TensorRTInfer>();
        
        // Find ArcFace engine file
        std::string arcface_engine_path;
        for (const auto& path : {"models/arcface/arcface.engine", 
                                "models/arcface/glintr100.onnx_b8_gpu0_fp16.engine",
                                "models/arcface/glintr100.onnx_b4_gpu0_fp16.engine"}) {
            if (fs::exists(path)) {
                arcface_engine_path = path;
                break;
            }
        }
        
        if (arcface_engine_path.empty()) {
            std::cerr << "⚠️ ArcFace engine not found - will test without embedding extraction" << std::endl;
        } else {
            if (tensorrtEngine->initialize(arcface_engine_path, "FP16")) {
                std::cout << "✅ TensorRT ArcFace engine loaded: " << arcface_engine_path << std::endl;
            } else {
                std::cerr << "⚠️ Failed to load TensorRT engine - will test without embedding extraction" << std::endl;
                tensorrtEngine.reset();
            }
        }

        // Set components in enrollment operations
        enrollOps.set_components(faceIndex, tensorrtEngine);

        // Get list of images in register folder
        std::string register_dir = "data/faces/register";
        std::vector<std::string> test_images;
        
        for (const auto& entry : fs::directory_iterator(register_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                    test_images.push_back(entry.path().string());
                }
            }
        }
        
        std::cout << "\nFound " << test_images.size() << " test images:" << std::endl;
        for (const auto& img_path : test_images) {
            std::cout << "  - " << fs::path(img_path).filename().string() << std::endl;
        }

        std::cout << "\n=== Testing Face Processing Pipeline ===" << std::endl;
        
        for (const auto& image_path : test_images) {
            std::string image_name = fs::path(image_path).stem().string();
            std::cout << "\n📸 Processing: " << image_name << std::endl;
            
            // Load the original image
            cv::Mat original_image = cv::imread(image_path);
            if (original_image.empty()) {
                std::cout << "❌ Failed to load image: " << image_path << std::endl;
                continue;
            }
            
            std::cout << "   Original size: " << original_image.cols << "x" << original_image.rows << std::endl;
            
            // Test face detection
            auto face_bbox = enrollOps.detect_face_bbox(original_image);
            if (!face_bbox.has_value()) {
                std::cout << "❌ No face detected in image" << std::endl;
                continue;
            }
            
            std::cout << "   ✅ Face detected at: [" << face_bbox->x << ", " << face_bbox->y 
                      << ", " << face_bbox->width << "x" << face_bbox->height << "]" << std::endl;
            
            // Test face alignment to 112x112
            auto aligned_face = enrollOps.crop_align_112(original_image);
            if (!aligned_face.has_value()) {
                std::cout << "❌ Failed to align face to 112x112" << std::endl;
                continue;
            }
            
            std::cout << "   ✅ Face aligned to: " << aligned_face->cols << "x" << aligned_face->rows << std::endl;
            
            // Save the aligned face for inspection
            std::string aligned_filename = aligned_output_dir + "/" + image_name + "_aligned_112x112.png";
            cv::imwrite(aligned_filename, *aligned_face);
            std::cout << "   💾 Saved aligned face: " << aligned_filename << std::endl;
            
            // Test blur assessment
            double blur_score = enrollOps.calculate_blur_variance(*aligned_face);
            std::cout << "   📊 Blur score: " << blur_score << (blur_score > 100.0 ? " (GOOD)" : " (BLURRY)") << std::endl;
            
            // Test embedding extraction
            auto embedding = enrollOps.extract_embedding(*aligned_face);
            if (!embedding.has_value()) {
                std::cout << "   ❌ Failed to extract face embedding" << std::endl;
                continue;
            }
            
            std::cout << "   ✅ Extracted embedding: " << embedding->size() << " dimensions" << std::endl;
            std::cout << "   📈 Embedding range: [" << *std::min_element(embedding->begin(), embedding->end()) 
                      << ", " << *std::max_element(embedding->begin(), embedding->end()) << "]" << std::endl;
            
            // Calculate embedding norm (should be ~1.0 after L2 normalization)
            double norm = 0.0;
            for (float val : *embedding) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            std::cout << "   🧮 Embedding L2 norm: " << norm << (std::abs(norm - 1.0) < 0.01 ? " (NORMALIZED)" : " (NOT NORMALIZED)") << std::endl;
        }

        std::cout << "\n🎉 Real face enrollment test completed successfully!" << std::endl;
        std::cout << "📁 Check aligned faces in: " << aligned_output_dir << std::endl;
        
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}