#include "enroll_ops.h"
#include "faiss_index.h"
#include "tensorrt_infer.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace EdgeDeepStream;

void analyze_alignment_quality(const cv::Mat& original, const cv::Mat& aligned, const std::string& name) {
    std::cout << "\n🔍 Analyzing " << name << " alignment quality:" << std::endl;
    
    // Check if aligned face is properly centered
    cv::Mat gray_aligned;
    cv::cvtColor(aligned, gray_aligned, cv::COLOR_BGR2GRAY);
    
    // Detect face in aligned image to verify centering
    cv::CascadeClassifier face_cascade;
    if (face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")) {
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray_aligned, faces, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20));
        
        if (!faces.empty()) {
            cv::Rect face = faces[0];
            int center_x = face.x + face.width / 2;
            int center_y = face.y + face.height / 2;
            int img_center_x = aligned.cols / 2;
            int img_center_y = aligned.rows / 2;
            
            double offset_x = abs(center_x - img_center_x);
            double offset_y = abs(center_y - img_center_y);
            
            std::cout << "   Face center: (" << center_x << "," << center_y << ")" << std::endl;
            std::cout << "   Image center: (" << img_center_x << "," << img_center_y << ")" << std::endl;
            std::cout << "   Centering offset: (" << offset_x << "," << offset_y << ")" << std::endl;
            std::cout << "   Face size in aligned: " << face.width << "x" << face.height << std::endl;
            
            // Calculate face-to-image ratio
            double face_ratio = (double)(face.width * face.height) / (aligned.cols * aligned.rows);
            std::cout << "   Face coverage: " << (face_ratio * 100) << "%" << std::endl;
            
            if (offset_x < 10 && offset_y < 10) {
                std::cout << "   ✅ Well-centered face" << std::endl;
            } else {
                std::cout << "   ⚠️ Face slightly off-center" << std::endl;
            }
            
            if (face_ratio > 0.3 && face_ratio < 0.8) {
                std::cout << "   ✅ Good face coverage ratio" << std::endl;
            } else if (face_ratio >= 0.8) {
                std::cout << "   ⚠️ Face very tight (high coverage)" << std::endl;
            } else {
                std::cout << "   ⚠️ Face too small (low coverage)" << std::endl;
            }
        } else {
            std::cout << "   ❌ No face detected in aligned image" << std::endl;
        }
    } else {
        std::cout << "   ⚠️ Could not load face cascade for analysis" << std::endl;
    }
}

void create_detailed_comparison(const cv::Mat& original, const cv::Mat& aligned, const std::string& output_path) {
    // Create a detailed comparison with annotations
    cv::Mat comparison;
    
    // Resize original for side-by-side comparison
    cv::Mat resized_original;
    int target_height = 224; // 2x the aligned face size for better visibility
    double scale = (double)target_height / original.rows;
    int new_width = static_cast<int>(original.cols * scale);
    cv::resize(original, resized_original, cv::Size(new_width, target_height));
    
    // Scale aligned face to same height
    cv::Mat scaled_aligned;
    cv::resize(aligned, scaled_aligned, cv::Size(224, 224)); // 2x scale for visibility
    
    // Create comparison canvas
    int canvas_width = resized_original.cols + scaled_aligned.cols + 30; // 30px spacing
    int canvas_height = std::max(resized_original.rows, scaled_aligned.rows) + 60; // 60px for labels
    cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    
    // Place images
    resized_original.copyTo(canvas(cv::Rect(10, 40, resized_original.cols, resized_original.rows)));
    scaled_aligned.copyTo(canvas(cv::Rect(resized_original.cols + 20, 40, scaled_aligned.cols, scaled_aligned.rows)));
    
    // Add labels and info
    cv::putText(canvas, "Original Image", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "Aligned 112x112 (2x scale)", cv::Point(resized_original.cols + 20, 25), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    // Add size info
    std::string orig_size = "Size: " + std::to_string(original.cols) + "x" + std::to_string(original.rows);
    cv::putText(canvas, orig_size, cv::Point(10, canvas_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    cv::putText(canvas, "Size: 112x112", cv::Point(resized_original.cols + 20, canvas_height - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    cv::imwrite(output_path, canvas);
}

int main() {
    try {
        std::cout << "=== Face Alignment Quality Validation ===" << std::endl;
        
        // Create output directories
        std::string output_dir = "test_output/alignment_validation";
        std::string aligned_dir = output_dir + "/aligned_faces";
        std::string comparison_dir = output_dir + "/comparisons";
        
        fs::create_directories(aligned_dir);
        fs::create_directories(comparison_dir);
        
        std::cout << "Output directories created:" << std::endl;
        std::cout << "  - " << aligned_dir << std::endl;
        std::cout << "  - " << comparison_dir << std::endl;
        
        // Initialize enrollment operations
        EnrollOps enrollOps;
        if (!enrollOps.initialize("config/config_pipeline.toml")) {
            std::cerr << "Failed to initialize enrollment operations" << std::endl;
            return 1;
        }
        
        // Test with register images
        std::vector<std::string> test_images;
        for (const auto& entry : fs::directory_iterator("data/faces/register")) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    test_images.push_back(entry.path().string());
                }
            }
        }
        
        std::cout << "\n=== Processing " << test_images.size() << " test images ===" << std::endl;
        
        for (const auto& img_path : test_images) {
            std::string img_name = fs::path(img_path).stem().string();
            std::cout << "\n📸 Processing: " << img_name << std::endl;
            
            // Load original image
            cv::Mat original = cv::imread(img_path);
            if (original.empty()) {
                std::cout << "❌ Failed to load: " << img_path << std::endl;
                continue;
            }
            
            std::cout << "   Original: " << original.cols << "x" << original.rows << std::endl;
            
            // Test face detection first
            auto bbox = enrollOps.detect_face_bbox(original);
            if (!bbox.has_value()) {
                std::cout << "   ❌ No face detected" << std::endl;
                continue;
            }
            
            std::cout << "   Face bbox: [" << bbox->x << "," << bbox->y << "," 
                     << bbox->width << "x" << bbox->height << "]" << std::endl;
            
            // Calculate face-to-image ratio in original
            double orig_face_ratio = (double)(bbox->width * bbox->height) / (original.cols * original.rows);
            std::cout << "   Original face coverage: " << (orig_face_ratio * 100) << "%" << std::endl;
            
            // Test alignment with default margin (0.10)
            auto aligned_default = enrollOps.crop_align_112(original, 0.10f);
            if (!aligned_default.has_value()) {
                std::cout << "   ❌ Failed to align with default margin" << std::endl;
                continue;
            }
            
            // Test with different margins
            auto aligned_tight = enrollOps.crop_align_112(original, 0.05f);  // Tighter
            auto aligned_loose = enrollOps.crop_align_112(original, 0.20f);  // Looser
            
            // Save aligned faces with different margins
            cv::imwrite(aligned_dir + "/" + img_name + "_margin_005.png", *aligned_tight);
            cv::imwrite(aligned_dir + "/" + img_name + "_margin_010.png", *aligned_default);
            cv::imwrite(aligned_dir + "/" + img_name + "_margin_020.png", *aligned_loose);
            
            std::cout << "   💾 Saved aligned faces with different margins" << std::endl;
            
            // Analyze quality for each margin setting
            analyze_alignment_quality(original, *aligned_tight, img_name + " (margin=0.05)");
            analyze_alignment_quality(original, *aligned_default, img_name + " (margin=0.10)");
            analyze_alignment_quality(original, *aligned_loose, img_name + " (margin=0.20)");
            
            // Create detailed comparison
            create_detailed_comparison(original, *aligned_default, 
                                     comparison_dir + "/" + img_name + "_comparison.png");
            
            std::cout << "   🔍 Created detailed comparison image" << std::endl;
        }
        
        std::cout << "\n🎉 Alignment quality validation completed!" << std::endl;
        std::cout << "📁 Check results in: " << output_dir << std::endl;
        std::cout << "\n📋 Key findings to verify:" << std::endl;
        std::cout << "  1. Face centering quality in aligned images" << std::endl;
        std::cout << "  2. Face coverage ratio (should be 30-80%)" << std::endl;
        std::cout << "  3. Margin effect on face tightness" << std::endl;
        std::cout << "  4. Overall alignment quality vs original" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}