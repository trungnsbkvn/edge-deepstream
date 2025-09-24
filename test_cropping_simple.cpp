#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

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
        
        // Draw the detected face on image for visual inspection
        cv::Mat image_with_face = image.clone();
        cv::rectangle(image_with_face, face, cv::Scalar(0, 255, 0), 3);
        cv::putText(image_with_face, "Original Detection", cv::Point(face.x, face.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        std::string face_debug_path = "test_output/face_detection_debug_" + name + ".png";
        cv::imwrite(face_debug_path, image_with_face);
        std::cout << "     💾 Saved debug image: " << face_debug_path << std::endl;
    }
}

cv::Rect improve_face_crop(const cv::Rect& original_face, const cv::Mat& image) {
    // Strategy: Reduce the face box size to focus more tightly on actual facial features
    // This helps remove background, shoulders, and clothing
    
    int new_width = static_cast<int>(original_face.width * 0.7);   // Reduce by 30%
    int new_height = static_cast<int>(original_face.height * 0.7); // Reduce by 30%
    
    // Keep the same center but make the box smaller
    int center_x = original_face.x + original_face.width / 2;
    int center_y = original_face.y + original_face.height / 2;
    
    // Move the center slightly up to focus more on face (eyes/nose area)
    center_y -= static_cast<int>(original_face.height * 0.08); // Move up by 8%
    
    int new_x = std::max(0, center_x - new_width / 2);
    int new_y = std::max(0, center_y - new_height / 2);
    
    // Ensure we don't go beyond image bounds
    new_x = std::min(new_x, image.cols - new_width);
    new_y = std::min(new_y, image.rows - new_height);
    new_width = std::min(new_width, image.cols - new_x);
    new_height = std::min(new_height, image.rows - new_y);
    
    return cv::Rect(new_x, new_y, new_width, new_height);
}

std::optional<cv::Mat> original_crop_align_112(const cv::Mat& bgr_image, float margin = 0.10f) {
    if (bgr_image.empty()) {
        return std::nullopt;
    }
    
    int H = bgr_image.rows;
    int W = bgr_image.cols;
    
    // Detect face using Haar cascade (original method)
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
        // Use the first face (original method)
        cv::Rect face = faces[0];
        
        // Apply margin and create crop (original method)
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
    
    // Fallback to center square
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
        if (coverage > 0.25) { // If face covers more than 25% of image, apply improvement
            std::cout << "   🔧 Applying improved cropping (face coverage: " << (coverage*100) << "%)" << std::endl;
            cv::Rect improved_face = improve_face_crop(face, bgr_image);
            
            // Debug: Draw both original and improved detection
            cv::Mat debug_image = bgr_image.clone();
            cv::rectangle(debug_image, face, cv::Scalar(0, 0, 255), 3); // Red = original
            cv::rectangle(debug_image, improved_face, cv::Scalar(0, 255, 0), 2); // Green = improved
            cv::putText(debug_image, "Red=Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            cv::putText(debug_image, "Green=Improved", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            std::string debug_path = "test_output/improved_detection_debug.png";
            cv::imwrite(debug_path, debug_image);
            std::cout << "     💾 Saved improved detection debug: " << debug_path << std::endl;
            
            face = improved_face;
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
        fs::create_directories("test_output");
        
        // Test specific problematic images
        std::vector<std::string> problem_images = {
            "data/faces/register/daniel.jpg",
            "data/faces/register/dien.jpg", 
            "data/faces/register/dien1.jpg"
        };
        
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
            auto original_aligned = original_crop_align_112(original);
            if (original_aligned.has_value()) {
                std::string orig_path = "test_output/" + img_name + "_original_crop.png";
                cv::imwrite(orig_path, *original_aligned);
                std::cout << "   💾 Saved original crop: " << orig_path << std::endl;
            }
            
            // Get improved alignment
            auto improved_aligned = improved_crop_align_112(original);
            if (improved_aligned.has_value()) {
                std::string imp_path = "test_output/" + img_name + "_improved_crop.png";
                cv::imwrite(imp_path, *improved_aligned);
                std::cout << "   💾 Saved improved crop: " << imp_path << std::endl;
                
                // Create side-by-side comparison
                if (original_aligned.has_value()) {
                    cv::Mat comparison;
                    cv::hconcat(*original_aligned, *improved_aligned, comparison);
                    
                    // Add labels
                    cv::putText(comparison, "Original", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                    cv::putText(comparison, "Improved", cv::Point(120, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                    
                    std::string comp_path = "test_output/" + img_name + "_comparison.png";
                    cv::imwrite(comp_path, comparison);
                    std::cout << "   🔍 Saved comparison: " << comp_path << std::endl;
                }
            }
        }
        
        std::cout << "\n🎉 Analysis completed!" << std::endl;
        std::cout << "📁 Check results in: test_output/" << std::endl;
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