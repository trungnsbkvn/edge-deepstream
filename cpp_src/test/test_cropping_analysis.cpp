#include "enroll_ops.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace EdgeDeepStream;

void analyze_face_detection(const cv::Mat& image, const std::string& name) {
    std::cout << "\n📸 Analyzing " << name << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    
    // Test Haar cascade detection
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")) {
        std::cout << "❌ Could not load face cascade" << std::endl;
        return;
    }
    
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(60, 60));
    
    std::cout << "   Found " << faces.size() << " faces:" << std::endl;
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Rect face = faces[i];
        double coverage = (double)(face.width * face.height) / (image.cols * image.rows) * 100;
        std::cout << "   Face " << i << ": [" << face.x << "," << face.y << "," 
                  << face.width << "x" << face.height << "] Coverage: " << coverage << "%" << std::endl;
                  
        // Check if face is too large (indicates poor detection)
        if (coverage > 35.0) {
            std::cout << "     ⚠️ Face detection too large - likely includes background/clothes" << std::endl;
        }
        if (face.width > image.cols * 0.6 || face.height > image.rows * 0.6) {
            std::cout << "     ⚠️ Face box too big relative to image size" << std::endl;
        }
    }
}

cv::Rect improve_face_crop(const cv::Rect& original_face, const cv::Mat& image) {
    // Strategy: Reduce the face box size to focus more tightly on actual facial features
    // This helps remove background, shoulders, and clothing
    
    int new_width = static_cast<int>(original_face.width * 0.75);   // Reduce by 25%
    int new_height = static_cast<int>(original_face.height * 0.75); // Reduce by 25%
    
    // Keep the same center but make the box smaller
    int center_x = original_face.x + original_face.width / 2;
    int center_y = original_face.y + original_face.height / 2;
    
    // Move the center slightly up to focus more on face (eyes/nose area)
    center_y -= static_cast<int>(original_face.height * 0.05); // Move up by 5%
    
    int new_x = std::max(0, center_x - new_width / 2);
    int new_y = std::max(0, center_y - new_height / 2);
    
    // Ensure we don't go beyond image bounds
    new_x = std::min(new_x, image.cols - new_width);
    new_y = std::min(new_y, image.rows - new_height);
    
    return cv::Rect(new_x, new_y, new_width, new_height);
}

std::optional<cv::Mat> improved_crop_align_112(const cv::Mat& bgr_image, float margin = 0.10f) {
    if (bgr_image.empty()) {
        return std::nullopt;
    }
    
    int H = bgr_image.rows;
    int W = bgr_image.cols;
    
    // Detect face using Haar cascade
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")) {
        return std::nullopt;
    }
    
    cv::Mat gray;
    cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(60, 60));
    
    cv::Mat crop;
    
    if (!faces.empty()) {
        // Use the largest face (assuming it's the main subject)
        cv::Rect face = *std::max_element(faces.begin(), faces.end(), 
                                         [](const cv::Rect& a, const cv::Rect& b) {
                                             return a.area() < b.area();
                                         });
        
        // Check if the detected face is suspiciously large (includes too much background)
        double coverage = (double)face.area() / (W * H);
        if (coverage > 0.35) { // If face covers more than 35% of image, it's probably too big
            std::cout << "   🔧 Applying improved cropping (face too large: " << (coverage*100) << "%)" << std::endl;
            face = improve_face_crop(face, bgr_image);
        }
        
        // Apply margin and create crop
        int x = face.x, y = face.y, w = face.width, h = face.height;
        int side0 = std::max(w, h);
        int side = static_cast<int>((1.0f + 2.0f * std::max(0.0f, margin)) * side0);
        int cx = x + w / 2;
        int cy = y + h / 2;
        int x0 = std::max(0, cx - side / 2);
        int y0 = std::max(0, cy - side / 2);
        int x1 = std::min(W, x0 + side);
        int y1 = std::min(H, y0 + side);
        
        cv::Rect crop_rect(x0, y0, x1 - x0, y1 - y0);
        if (crop_rect.width > 0 && crop_rect.height > 0) {
            crop = bgr_image(crop_rect).clone();
        }
    }
    
    // Fallback to center square if no good face detected
    if (crop.empty()) {
        int side = std::min(H, W);
        int cx = W / 2, cy = H / 2;
        int x0 = std::max(0, cx - side / 2);
        int y0 = std::max(0, cy - side / 2);
        cv::Rect crop_rect(x0, y0, side, side);
        crop = bgr_image(crop_rect).clone();
    }
    
    if (crop.empty()) {
        return std::nullopt;
    }
    
    try {
        cv::Mat face112;
        cv::resize(crop, face112, cv::Size(112, 112), 0, 0, cv::INTER_LINEAR);
        return face112;
    } catch (const std::exception& e) {
        return std::nullopt;
    }
}

int main() {
    try {
        std::cout << "=== Face Cropping Analysis & Improvement ===" << std::endl;
        
        // Create output directory
        std::string output_dir = "test_output/improved_cropping";
        fs::create_directories(output_dir + "/original");
        fs::create_directories(output_dir + "/improved");
        fs::create_directories(output_dir + "/comparison");
        
        // Test specific problematic images
        std::vector<std::string> problem_images = {
            "data/faces/register/daniel.jpg",
            "data/faces/register/dien.jpg", 
            "data/faces/register/dien1.jpg"
        };
        
        // Initialize enrollment operations for comparison
        EnrollOps enrollOps;
        enrollOps.initialize("config/config_pipeline.toml");
        
        for (const auto& img_path : problem_images) {
            std::string img_name = fs::path(img_path).stem().string();
            
            // Load original image
            cv::Mat original = cv::imread(img_path);
            if (original.empty()) {
                std::cout << "❌ Failed to load: " << img_path << std::endl;
                continue;
            }
            
            // Analyze current face detection
            analyze_face_detection(original, img_name);
            
            // Get original alignment
            auto original_aligned = enrollOps.crop_align_112(original);
            if (original_aligned.has_value()) {
                cv::imwrite(output_dir + "/original/" + img_name + "_original_crop.png", *original_aligned);
                std::cout << "   💾 Saved original crop" << std::endl;
            }
            
            // Get improved alignment
            auto improved_aligned = improved_crop_align_112(original);
            if (improved_aligned.has_value()) {
                cv::imwrite(output_dir + "/improved/" + img_name + "_improved_crop.png", *improved_aligned);
                std::cout << "   💾 Saved improved crop" << std::endl;
                
                // Create side-by-side comparison
                if (original_aligned.has_value()) {
                    cv::Mat comparison;
                    cv::hconcat(*original_aligned, *improved_aligned, comparison);
                    
                    // Add labels
                    cv::putText(comparison, "Original", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                    cv::putText(comparison, "Improved", cv::Point(120, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                    
                    cv::imwrite(output_dir + "/comparison/" + img_name + "_comparison.png", comparison);
                    std::cout << "   🔍 Saved comparison" << std::endl;
                }
            }
        }
        
        std::cout << "\n🎉 Analysis completed!" << std::endl;
        std::cout << "📁 Check results in: " << output_dir << std::endl;
        std::cout << "\n📋 Expected improvements:" << std::endl;
        std::cout << "  - Less background in cropped faces" << std::endl;
        std::cout << "  - Tighter focus on facial features" << std::endl;
        std::cout << "  - Reduced clothing/shoulder inclusion" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}