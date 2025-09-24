#include "enroll_ops.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace EdgeDeepStream;

void analyze_specific_image(const std::string& image_path, EnrollOps& enrollOps) {
    std::string image_name = fs::path(image_path).stem().string();
    std::cout << "\n🔍 Analyzing: " << image_name << std::endl;
    
    cv::Mat original = cv::imread(image_path);
    if (original.empty()) {
        std::cout << "❌ Failed to load image" << std::endl;
        return;
    }
    
    std::cout << "   Original size: " << original.cols << "x" << original.rows << std::endl;
    
    // Get face detection
    auto bbox = enrollOps.detect_face_bbox(original);
    if (!bbox.has_value()) {
        std::cout << "❌ No face detected" << std::endl;
        return;
    }
    
    // Calculate original face coverage
    double face_coverage = (double)(bbox->width * bbox->height) / (original.cols * original.rows);
    std::cout << "   Face bbox: [" << bbox->x << "," << bbox->y << "," 
             << bbox->width << "x" << bbox->height << "]" << std::endl;
    std::cout << "   Face coverage: " << (face_coverage * 100) << "%" << std::endl;
    
    // Test different cropping approaches
    std::vector<float> margins = {0.05f, 0.10f, 0.15f};
    std::string output_dir = "test_output/crop_analysis/" + image_name;
    fs::create_directories(output_dir);
    
    for (float margin : margins) {
        auto aligned = enrollOps.crop_align_112(original, margin);
        if (aligned.has_value()) {
            std::string filename = output_dir + "/margin_" + 
                                 std::to_string(static_cast<int>(margin * 100)) + ".png";
            cv::imwrite(filename, *aligned);
            std::cout << "   💾 Saved margin " << margin << ": " << filename << std::endl;
        }
    }
    
    // Create annotated original with face box
    cv::Mat annotated = original.clone();
    cv::rectangle(annotated, *bbox, cv::Scalar(0, 255, 0), 3);
    cv::putText(annotated, "Detected Face", 
                cv::Point(bbox->x, bbox->y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    cv::imwrite(output_dir + "/original_with_detection.png", annotated);
    std::cout << "   🎯 Saved annotated original" << std::endl;
}

int main() {
    try {
        std::cout << "=== Specific Image Crop Analysis ===" << std::endl;
        
        // Initialize enrollment operations
        EnrollOps enrollOps;
        if (!enrollOps.initialize("config/config_pipeline.toml")) {
            std::cerr << "Failed to initialize enrollment operations" << std::endl;
            return 1;
        }
        
        // Analyze the specific problematic images
        std::vector<std::string> problem_images = {
            "data/faces/register/daniel.jpg",
            "data/faces/register/dien.jpg", 
            "data/faces/register/dien1.jpg"
        };
        
        for (const auto& img_path : problem_images) {
            if (fs::exists(img_path)) {
                analyze_specific_image(img_path, enrollOps);
            } else {
                std::cout << "⚠️ Image not found: " << img_path << std::endl;
            }
        }
        
        std::cout << "\n🎉 Crop analysis completed!" << std::endl;
        std::cout << "📁 Check results in: test_output/crop_analysis/" << std::endl;
        std::cout << "\n📋 Improvements made:" << std::endl;
        std::cout << "  1. More aggressive cropping for large face detections" << std::endl;
        std::cout << "  2. Reduced margin expansion for large faces" << std::endl;
        std::cout << "  3. Better vertical centering to preserve mouth area" << std::endl;
        std::cout << "  4. Three-tier cropping strategy based on face size" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Analysis failed: " << e.what() << std::endl;
        return 1;
    }
}