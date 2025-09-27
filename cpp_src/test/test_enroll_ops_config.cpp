#include "enroll_ops.h"
#include "faiss_index.h"
#include "config_parser.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <random>

using namespace EdgeDeepStream;

// Helper function to create test directories
void setupTestDirectories(const Config& config) {
    std::filesystem::create_directories(config.register_folder);
    std::filesystem::create_directories(config.recognized_folder);
    std::filesystem::create_directories(config.aligned_folder);
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
        
        Config config;
        if (!parseConfig(config_path, config)) {
            std::cerr << "✗ Failed to parse config file" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Configuration loaded successfully" << std::endl;
        std::cout << "  - FAISS index path: " << config.faiss_index_path << std::endl;
        std::cout << "  - FAISS labels path: " << config.faiss_labels_path << std::endl;
        std::cout << "  - Faces data path: " << config.faces_data_path << std::endl;
        
        // Test 2: Setup test environment
        std::cout << "\n--- Test 2: Setup test environment ---" << std::endl;
        setupTestDirectories(config);
        std::cout << "✓ Test directories created" << std::endl;
        
        // Test 3: Create FAISS index
        std::cout << "\n--- Test 3: Create FAISS index ---" << std::endl;
        FaceIndex::Config faiss_config;
        faiss_config.metric = "cosine";
        faiss_config.index_type = "flat";
        faiss_config.use_gpu = false;
        
        auto faiss_index = std::make_shared<FaceIndex>(512, faiss_config);
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
        cv::Mat test_image = createTestImage();
        std::vector<float> test_feature = createTestFeature();
        
        // Save test image to register folder
        std::string image_path = config.register_folder + "/" + person_id + ".jpg";
        cv::imwrite(image_path, test_image);
        std::cout << "✓ Test image saved: " << image_path << std::endl;
        
        // Test add person
        auto result = enroll_ops->addPerson(person_id, test_feature);
        if (result.success) {
            std::cout << "✓ Person added successfully: " << result.message << std::endl;
        } else {
            std::cout << "✗ Failed to add person: " << result.message << std::endl;
        }
        
        // Test 6: Test person deletion
        std::cout << "\n--- Test 6: Test person deletion ---" << std::endl;
        auto delete_result = enroll_ops->deletePerson(person_id);
        if (delete_result.success) {
            std::cout << "✓ Person deleted successfully: " << delete_result.message << std::endl;
        } else {
            std::cout << "✗ Failed to delete person: " << delete_result.message << std::endl;
        }
        
        // Test 7: Test error handling
        std::cout << "\n--- Test 7: Test error handling ---" << std::endl;
        
        // Try to delete non-existent person
        auto error_result = enroll_ops->deletePerson("non_existent_person");
        if (!error_result.success) {
            std::cout << "✓ Error handling works: " << error_result.message << std::endl;
        }
        
        // Try to add person with empty feature
        std::vector<float> empty_feature;
        auto empty_result = enroll_ops->addPerson("test_empty", empty_feature);
        if (!empty_result.success) {
            std::cout << "✓ Empty feature validation works: " << empty_result.message << std::endl;
        }
        
        // Test 8: Check config values
        std::cout << "\n--- Test 8: Check config integration ---" << std::endl;
        std::cout << "✓ Using actual config paths:" << std::endl;
        std::cout << "  - Register folder: " << config.register_folder << std::endl;
        std::cout << "  - Recognized folder: " << config.recognized_folder << std::endl;
        std::cout << "  - Aligned folder: " << config.aligned_folder << std::endl;
        
        std::cout << "\n=== All tests completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}