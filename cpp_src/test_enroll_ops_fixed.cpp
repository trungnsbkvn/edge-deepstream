#include "enroll_ops.h"
#include "config_parser.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <random>

using namespace EdgeDeepStream;

// Helper function to create test directories based on config
void setupTestDirectories(const Config& config) {
    // Get directories from config
    std::string known_faces_dir = config.get<std::string>("pipeline", "known_face_dir", "data/known_faces");
    std::string recognized_dir = config.get<std::string>("recognition", "save_dir", "data/faces/recognized");
    
    // Create additional test directories
    std::string register_dir = known_faces_dir + "/register";
    std::string aligned_dir = known_faces_dir + "/aligned";
    
    std::filesystem::create_directories(register_dir);
    std::filesystem::create_directories(recognized_dir);
    std::filesystem::create_directories(aligned_dir);
    std::filesystem::create_directories(known_faces_dir);
    
    std::cout << "Created directories:" << std::endl;
    std::cout << "  - Known faces: " << known_faces_dir << std::endl;
    std::cout << "  - Register: " << register_dir << std::endl;
    std::cout << "  - Recognized: " << recognized_dir << std::endl;
    std::cout << "  - Aligned: " << aligned_dir << std::endl;
}

// Helper function to create a test image
cv::Mat createTestImage(int width = 112, int height = 112) {
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
    // Create a simple pattern
    cv::rectangle(img, cv::Point(10, 10), cv::Point(width-10, height-10), cv::Scalar(100, 150, 200), -1);
    cv::circle(img, cv::Point(width/2, height/2), 20, cv::Scalar(255, 255, 255), -1);
    return img;
}

// Helper function to create test feature vector
std::vector<float> createTestFeature(int size = 512) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    std::vector<float> feature(size);
    for (int i = 0; i < size; ++i) {
        feature[i] = dis(gen);
    }
    return feature;
}

int main() {
    std::cout << "=== EnrollOps Test Program (with Config) ===" << std::endl;
    
    try {
        // Test 1: Load configuration
        std::cout << "\n--- Test 1: Load configuration ---" << std::endl;
        
        std::string config_path = "/home/m2n/edge-deepstream/config/config_pipeline.toml";
        if (!std::filesystem::exists(config_path)) {
            std::cerr << "✗ Config file not found: " << config_path << std::endl;
            return 1;
        }
        
        auto config_parser = std::make_unique<ConfigParser>();
        auto config = config_parser->parse_toml(config_path);
        if (!config) {
            std::cerr << "✗ Failed to parse config file" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Configuration loaded successfully" << std::endl;
        std::cout << "  - FAISS index path: " << config->get<std::string>("recognition", "index_path", "N/A") << std::endl;
        std::cout << "  - FAISS labels path: " << config->get<std::string>("recognition", "labels_path", "N/A") << std::endl;
        std::cout << "  - Known faces dir: " << config->get<std::string>("pipeline", "known_face_dir", "N/A") << std::endl;
        
        // Test 2: Setup test environment
        std::cout << "\n--- Test 2: Setup test environment ---" << std::endl;
        setupTestDirectories(*config);
        std::cout << "✓ Test directories created" << std::endl;
        
        // Test 3: Create FAISS index
        std::cout << "\n--- Test 3: Create FAISS index ---" << std::endl;
        FaissIndex::Config faiss_config;
        faiss_config.metric = config->get<std::string>("recognition", "metric", "cosine");
        faiss_config.index_type = config->get<std::string>("recognition", "index_type", "flat");
        faiss_config.use_gpu = config->get<bool>("recognition", "use_gpu", false);
        
        auto faiss_index = std::make_shared<FaissIndex>(512, faiss_config);
        if (faiss_index) {
            std::cout << "✓ FAISS index created successfully" << std::endl;
        }
        
        // Test 4: Create EnrollOps with actual config
        std::cout << "\n--- Test 4: Create EnrollOps ---" << std::endl;
        auto enroll_ops = std::make_unique<EnrollOps>();
        
        if (!enroll_ops->initialize(config_path)) {
            std::cerr << "✗ Failed to initialize EnrollOps" << std::endl;
            return 1;
        }
        
        // Set components
        enroll_ops->set_components(faiss_index, nullptr);  // No TensorRT for this test
        std::cout << "✓ EnrollOps created and initialized successfully" << std::endl;
        
        // Test 5: Test person enrollment
        std::cout << "\n--- Test 5: Test person enrollment ---" << std::endl;
        
        // Create test data
        std::string person_id = "test_person_001";
        std::string person_name = "Test User 001";
        cv::Mat test_image = createTestImage();
        std::vector<float> test_feature = createTestFeature();
        
        // Save test image to known faces dir (like what would happen in real enrollment)
        std::string known_faces_dir = config->get<std::string>("pipeline", "known_face_dir", "data/known_faces");
        std::string image_path = known_faces_dir + "/" + person_id + ".jpg";
        cv::imwrite(image_path, test_image);
        std::cout << "✓ Test image saved: " << image_path << std::endl;
        
        // Test enroll person
        auto result = enroll_ops->enroll_person(person_id, person_name, test_image, test_feature);
        if (result.success) {
            std::cout << "✓ Person enrolled successfully: " << result.message << std::endl;
        } else {
            std::cout << "✗ Failed to enroll person: " << result.message << std::endl;
        }
        
        // Test 6: Test person deletion
        std::cout << "\n--- Test 6: Test person deletion ---" << std::endl;
        auto delete_result = enroll_ops->delete_person(person_id, true);  // remove_files = true
        if (delete_result.success) {
            std::cout << "✓ Person deleted successfully: " << delete_result.message << std::endl;
        } else {
            std::cout << "✗ Failed to delete person: " << delete_result.message << std::endl;
        }
        
        // Test 7: Test error handling
        std::cout << "\n--- Test 7: Test error handling ---" << std::endl;
        
        // Try to delete non-existent person
        auto error_result = enroll_ops->delete_person("non_existent_person", false);
        if (!error_result.success) {
            std::cout << "✓ Error handling works: " << error_result.message << std::endl;
        }
        
        // Try to enroll person with empty feature
        std::vector<float> empty_feature;
        cv::Mat empty_image;
        auto empty_result = enroll_ops->enroll_person("test_empty", "Test Empty", empty_image, empty_feature);
        if (!empty_result.success) {
            std::cout << "✓ Empty feature validation works: " << empty_result.message << std::endl;
        }
        
        // Test 8: Check config integration
        std::cout << "\n--- Test 8: Check config integration ---" << std::endl;
        std::cout << "✓ Using actual config values:" << std::endl;
        std::cout << "  - Known faces dir: " << config->get<std::string>("pipeline", "known_face_dir", "N/A") << std::endl;
        std::cout << "  - Recognition threshold: " << config->get<std::string>("recognition", "threshold", "0.5") << std::endl;
        std::cout << "  - FAISS backend: " << config->get<std::string>("recognition", "backend", "faiss") << std::endl;
        std::cout << "  - Index type: " << config->get<std::string>("recognition", "index_type", "flat") << std::endl;
        
        std::cout << "\n=== All tests completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}