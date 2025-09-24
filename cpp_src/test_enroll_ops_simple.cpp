#include "enroll_ops.h"
#include "faiss_index.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <random>

using namespace EdgeDeepStream;

// Mock Config struct for testing
struct MockConfig {
    bool deepstream_recognition_enabled = true;
    std::string faiss_index_path = "/tmp/test.index";
    std::string faiss_labels_path = "/tmp/labels.json";
    std::string faces_data_path = "/tmp/faces";
    std::string register_folder = "/tmp/faces/register";
    std::string recognized_folder = "/tmp/faces/recognized";
    std::string aligned_folder = "/tmp/faces/aligned";
};

// Helper function to create test directories
void setupTestDirectories(const MockConfig& config) {
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
    std::cout << "=== EnrollOps Test Program ===" << std::endl;
    
    try {
        // Test 1: Setup
        std::cout << "\n--- Test 1: Setup test environment ---" << std::endl;
        MockConfig config;
        setupTestDirectories(config);
        std::cout << "✓ Test directories created" << std::endl;
        
        // Test 2: Create FAISS index
        std::cout << "\n--- Test 2: Create FAISS index ---" << std::endl;
        FaissIndex::Config faiss_config;
        faiss_config.metric = "cosine";
        faiss_config.index_type = "flat";
        faiss_config.use_gpu = false;
        
        auto faiss_index = std::make_shared<FaissIndex>(512, faiss_config);
        if (faiss_index) {
            std::cout << "✓ FAISS index created successfully" << std::endl;
        }
        
        // Test 3: Create EnrollOps without config dependency
        std::cout << "\n--- Test 3: Create EnrollOps ---" << std::endl;
        auto enroll_ops = std::make_unique<EnrollOps>(
            faiss_index, 
            nullptr,  // No TensorRT for this test
            config.faiss_index_path,
            config.faiss_labels_path,
            config.faces_data_path
        );
        
        if (enroll_ops) {
            std::cout << "✓ EnrollOps created successfully" << std::endl;
        }
        
        // Test 4: Test person enrollment (basic)
        std::cout << "\n--- Test 4: Test person enrollment ---" << std::endl;
        
        // Create test data
        std::string person_id = "test_person_001";
        cv::Mat test_image = createTestImage();
        std::vector<float> test_feature = createTestFeature();
        
        // Save test image
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
        
        // Test 5: Test person deletion
        std::cout << "\n--- Test 5: Test person deletion ---" << std::endl;
        auto delete_result = enroll_ops->deletePerson(person_id);
        if (delete_result.success) {
            std::cout << "✓ Person deleted successfully: " << delete_result.message << std::endl;
        } else {
            std::cout << "✗ Failed to delete person: " << delete_result.message << std::endl;
        }
        
        // Test 6: Test error handling
        std::cout << "\n--- Test 6: Test error handling ---" << std::endl;
        
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
        
        std::cout << "\n=== All tests completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}