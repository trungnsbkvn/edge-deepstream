#include <iostream>
#include <memory>

// Just test basic class instantiation
namespace EdgeDeepStream {
    class EnrollOps {
    public:
        EnrollOps() { std::cout << "EnrollOps constructor called" << std::endl; }
        ~EnrollOps() { std::cout << "EnrollOps destructor called" << std::endl; }
        
        bool initialize() {
            std::cout << "EnrollOps initialized" << std::endl;
            return true;
        }
    };
}

using namespace EdgeDeepStream;

int main() {
    std::cout << "=== Minimal EnrollOps Test ===" << std::endl;
    
    try {
        // Test 1: Basic instantiation
        std::cout << "\n--- Test 1: Basic instantiation ---" << std::endl;
        auto enroll_ops = std::make_unique<EnrollOps>();
        std::cout << "✓ EnrollOps created successfully" << std::endl;
        
        // Test 2: Basic initialization
        std::cout << "\n--- Test 2: Basic initialization ---" << std::endl;
        if (enroll_ops->initialize()) {
            std::cout << "✓ EnrollOps initialized successfully" << std::endl;
        }
        
        std::cout << "\n=== Minimal test completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}