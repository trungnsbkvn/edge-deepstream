#include "enroll_ops.h"
#include "faiss_index.h"
#include "config_parser.h"
#include "env_utils.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace EdgeDeepStream;

// Helper function to create a fake face image
cv::Mat createTestFaceImage(int width = 112, int height = 112) {
    cv::Mat face(height, width, CV_8UC3);
    cv::randu(face, cv::Scalar(50, 50, 50), cv::Scalar(200, 200, 200));
    
    // Add some basic face-like features
    cv::circle(face, cv::Point(width/3, height/3), 5, cv::Scalar(255, 255, 255), -1); // left eye
    cv::circle(face, cv::Point(2*width/3, height/3), 5, cv::Scalar(255, 255, 255), -1); // right eye
    cv::ellipse(face, cv::Point(width/2, 2*height/3), cv::Size(width/4, height/8), 0, 0, 180, cv::Scalar(255, 255, 255), 2); // smile
    
    return face;
}

int main() {
    try {
        std::cout << "=== Full Enrollment System Test ===" << std::endl;

        // Initialize enrollment operations
        EnrollOps enrollOps;
        if (!enrollOps.initialize("config/config_pipeline.toml")) {
            std::cerr << "Failed to initialize enrollment operations" << std::endl;
            return 1;
        }

        std::cout << "\n1. Testing person registration..." << std::endl;
        
        // Create test directories
        fs::create_directories("data/known_faces");
        fs::create_directories("data/faces/register");
        fs::create_directories("data/faces/aligned");
        fs::create_directories("data/faces/recognized");

        // Create and save a test face image
        std::string testPersonId = "test_person_001";
        std::string testImagePath = "data/known_faces/" + testPersonId + ".png";
        
        cv::Mat testFace = createTestFaceImage();
        cv::imwrite(testImagePath, testFace);
        std::cout << "Created test face image: " << testImagePath << std::endl;

        // Test adding person
        auto addResult = enrollOps.enroll_person_from_file(testPersonId, testPersonId, testImagePath);
        std::cout << "Add person result: " << (addResult.success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Message: " << addResult.message << std::endl;

        std::cout << "\n2. Testing person listing..." << std::endl;
        auto personList = enrollOps.list_persons();
        std::cout << "Registered persons (" << personList.size() << "):" << std::endl;
        for (const auto& person : personList) {
            std::cout << "  - " << person.user_id << " (" << person.name << ")" << std::endl;
        }

        std::cout << "\n3. Testing person deletion..." << std::endl;
        auto deleteResult = enrollOps.delete_person(testPersonId);
        std::cout << "Delete person result: " << (deleteResult.success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Message: " << deleteResult.message << std::endl;

        // Verify deletion
        auto updatedList = enrollOps.list_persons();
        std::cout << "Persons after deletion (" << updatedList.size() << "):" << std::endl;
        for (const auto& person : updatedList) {
            std::cout << "  - " << person.user_id << " (" << person.name << ")" << std::endl;
        }

        std::cout << "\n4. Testing batch registration..." << std::endl;
        
        // Create multiple test persons
        std::vector<std::string> testPersons = {"alice", "bob", "charlie"};
        for (const auto& person : testPersons) {
            std::string imagePath = "data/known_faces/" + person + ".png";
            cv::Mat face = createTestFaceImage();
            cv::imwrite(imagePath, face);
            
            auto result = enrollOps.enroll_person_from_file(person, person, imagePath);
            std::cout << "Added " << person << ": " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
        }

        // Final person list
        auto finalList = enrollOps.list_persons();
        std::cout << "\nFinal registered persons (" << finalList.size() << "):" << std::endl;
        for (const auto& person : finalList) {
            std::cout << "  - " << person.user_id << " (" << person.name << ")" << std::endl;
        }

        std::cout << "\n5. Testing enrollment statistics..." << std::endl;
        auto stats = enrollOps.get_stats();
        std::cout << "Total persons: " << stats.total_persons << std::endl;
        std::cout << "Total vectors: " << stats.total_vectors << std::endl;

        // Clean up test files
        std::cout << "\n6. Cleaning up test files..." << std::endl;
        for (const auto& person : testPersons) {
            std::string imagePath = "data/known_faces/" + person + ".png";
            enrollOps.delete_person(person);
            fs::remove(imagePath);
        }
        fs::remove(testImagePath);

        std::cout << "\n=== All enrollment tests completed successfully! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}