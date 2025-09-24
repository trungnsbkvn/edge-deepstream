#include "enroll_ops.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace EdgeDeepStream;

void debug_bbox_usage(const std::string& image_path, EnrollOps& enrollOps) {
    std::string image_name = fs::path(image_path).stem().string();
    std::cout << "\n🔍 Debug analysis: " << image_name << std::endl;
    
    cv::Mat original = cv::imread(image_path);
    if (original.empty()) {
        std::cout << "❌ Failed to load image" << std::endl;
        return;
    }
    
    std::cout << "   Original size: " << original.cols << "x" << original.rows << std::endl;
    
    // Get face detection (same method used by crop_align_112)
    auto bbox = enrollOps.detect_face_bbox(original);
    if (!bbox.has_value()) {
        std::cout << "❌ No face detected" << std::endl;
        return;
    }
    
    int H = original.rows;
    int W = original.cols;
    int x = bbox->x, y = bbox->y, w = bbox->width, h = bbox->height;
    
    std::cout << "   Detected bbox: [" << x << "," << y << "," << w << "x" << h << "]" << std::endl;
    
    // Show the current crop_align_112 logic step by step
    std::cout << "\n   Current crop_align_112 logic:" << std::endl;
    int side = std::max(w, h);
    int cx = x + w / 2;
    int cy = y + h / 2;
    int x0 = std::max(0, cx - side / 2);
    int y0 = std::max(0, cy - side / 2);
    int x1 = std::min(W, x0 + side);
    int y1 = std::min(H, y0 + side);
    
    std::cout << "   Side length: " << side << std::endl;
    std::cout << "   Face center: (" << cx << "," << cy << ")" << std::endl;
    std::cout << "   Crop coords: x0=" << x0 << ", y0=" << y0 << ", x1=" << x1 << ", y1=" << y1 << std::endl;
    std::cout << "   Final crop rect: [" << x0 << "," << y0 << "," << (x1-x0) << "x" << (y1-y0) << "]" << std::endl;
    
    // Show direct bbox usage (crop analysis approach)
    std::cout << "\n   Direct bbox usage (crop analysis approach):" << std::endl;
    std::cout << "   Direct crop rect: [" << x << "," << y << "," << w << "x" << h << "]" << std::endl;
    
    // Create both crops and save them
    cv::Rect current_crop(x0, y0, x1 - x0, y1 - y0);
    cv::Rect direct_crop(*bbox);
    
    std::string output_dir = "test_output/bbox_debug/" + image_name;
    fs::create_directories(output_dir);
    
    // Current implementation crop
    if (current_crop.width > 0 && current_crop.height > 0) {
        cv::Mat crop_current = original(current_crop).clone();
        cv::Mat face112_current;
        cv::resize(crop_current, face112_current, cv::Size(112, 112));
        cv::imwrite(output_dir + "/current_method.png", face112_current);
        std::cout << "   💾 Saved current method result" << std::endl;
    }
    
    // Direct bbox crop 
    if (direct_crop.width > 0 && direct_crop.height > 0) {
        cv::Mat crop_direct = original(direct_crop).clone();
        cv::Mat face112_direct;
        cv::resize(crop_direct, face112_direct, cv::Size(112, 112));
        cv::imwrite(output_dir + "/direct_bbox.png", face112_direct);
        std::cout << "   💾 Saved direct bbox result" << std::endl;
    }
    
    // Create visual comparison
    cv::Mat annotated = original.clone();
    
    // Draw current method crop in red
    cv::rectangle(annotated, current_crop, cv::Scalar(0, 0, 255), 3);  // Red
    cv::putText(annotated, "Current Method", 
                cv::Point(current_crop.x, current_crop.y - 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    
    // Draw direct bbox in green
    cv::rectangle(annotated, direct_crop, cv::Scalar(0, 255, 0), 2);   // Green
    cv::putText(annotated, "Direct BBox", 
                cv::Point(direct_crop.x, direct_crop.y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    cv::imwrite(output_dir + "/bbox_comparison.png", annotated);
    std::cout << "   🎯 Saved bbox comparison" << std::endl;
}

int main() {
    try {
        std::cout << "=== BBox Usage Debug Analysis ===" << std::endl;
        
        // Initialize enrollment operations
        EnrollOps enrollOps;
        if (!enrollOps.initialize("config/config_pipeline.toml")) {
            std::cerr << "Failed to initialize enrollment operations" << std::endl;
            return 1;
        }
        
        // Debug the specific images
        std::vector<std::string> images = {
            "data/faces/register/daniel.jpg",
            "data/faces/register/dien.jpg", 
            "data/faces/register/dien1.jpg"
        };
        
        for (const auto& img_path : images) {
            if (fs::exists(img_path)) {
                debug_bbox_usage(img_path, enrollOps);
            } else {
                std::cout << "⚠️ Image not found: " << img_path << std::endl;
            }
        }
        
        std::cout << "\n🎉 Debug analysis completed!" << std::endl;
        std::cout << "📁 Check results in: test_output/bbox_debug/" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Analysis failed: " << e.what() << std::endl;
        return 1;
    }
}