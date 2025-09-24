#include "faiss_index.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace EdgeDeepStream;

// Generate random normalized feature vector (simulating face embeddings)
std::vector<float> generate_random_feature(int dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> feature(dim);
    
    for (int i = 0; i < dim; ++i) {
        feature[i] = dist(rng);
    }
    
    // Normalize for cosine similarity
    float norm = 0.0f;
    for (float v : feature) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-12f) {
        for (float& v : feature) {
            v /= norm;
        }
    }
    
    return feature;
}

int main() {
    std::cout << "=== EdgeDeepStream FAISS Integration Test ===" << std::endl;
    
#ifdef HAVE_FAISS
    std::cout << "✅ FAISS support enabled" << std::endl;
#else
    std::cout << "❌ FAISS support not available" << std::endl;
    return 1;
#endif

    try {
        // Test configuration
        const int feature_dim = 512;  // ArcFace standard dimension
        const int num_test_faces = 10;
        
        // Create face index with cosine similarity
        FaceIndex::Config config("cosine", "flat");
        FaceIndex face_index(feature_dim, config);
        
        std::cout << "\n=== Test 1: Adding Face Features ===" << std::endl;
        
        // Generate test data
        std::mt19937 rng(42);  // Fixed seed for reproducible results
        std::vector<std::string> names;
        std::vector<std::vector<float>> features;
        
        for (int i = 0; i < num_test_faces; ++i) {
            names.push_back("Person_" + std::to_string(i));
            features.push_back(generate_random_feature(feature_dim, rng));
        }
        
        // Add features to index
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = face_index.add(names, features);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "✅ Added " << num_test_faces << " faces in " << duration.count() << " μs" << std::endl;
            std::cout << "   Index size: " << face_index.size() << " faces" << std::endl;
        } else {
            std::cout << "❌ Failed to add faces to index" << std::endl;
            return 1;
        }
        
        std::cout << "\n=== Test 2: Face Recognition Search ===" << std::endl;
        
        // Test search with exact matches
        for (int i = 0; i < std::min(3, num_test_faces); ++i) {
            std::cout << "\nSearching for: " << names[i] << std::endl;
            
            start_time = std::chrono::high_resolution_clock::now();
            auto result = face_index.search_top1(features[i]);
            end_time = std::chrono::high_resolution_clock::now();
            
            auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "  Result: " << result.first << " (score: " << result.second << ")" << std::endl;
            std::cout << "  Search time: " << search_duration.count() << " μs" << std::endl;
            
            if (result.first == names[i]) {
                std::cout << "  ✅ Exact match found!" << std::endl;
            } else {
                std::cout << "  ⚠️  Different person matched" << std::endl;
            }
        }
        
        std::cout << "\n=== Test 3: Top-K Search ===" << std::endl;
        
        // Test k-nearest neighbors search
        const int k = 3;
        auto query_feature = features[0];  // Use first person as query
        
        start_time = std::chrono::high_resolution_clock::now();
        auto knn_results = face_index.search(query_feature, k);
        end_time = std::chrono::high_resolution_clock::now();
        
        auto knn_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Top-" << k << " matches for " << names[0] << ":" << std::endl;
        for (size_t i = 0; i < knn_results.size(); ++i) {
            int idx = knn_results[i].first;
            float score = knn_results[i].second;
            if (idx >= 0 && idx < static_cast<int>(names.size())) {
                std::cout << "  " << (i+1) << ". " << names[idx] << " (score: " << score << ")" << std::endl;
            }
        }
        std::cout << "Search time: " << knn_duration.count() << " μs" << std::endl;
        
        std::cout << "\n=== Test 4: Save and Load Index ===" << std::endl;
        
        // Test saving
        std::string index_path = "/tmp/test_face.index";
        std::string labels_path = "/tmp/test_labels.json";
        
        start_time = std::chrono::high_resolution_clock::now();
        bool save_success = face_index.save(index_path, labels_path);
        end_time = std::chrono::high_resolution_clock::now();
        
        if (save_success) {
            auto save_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "✅ Index saved in " << save_duration.count() << " μs" << std::endl;
            
            // Test loading
            FaceIndex loaded_index(feature_dim, config);
            
            start_time = std::chrono::high_resolution_clock::now();
            bool load_success = loaded_index.load(index_path, labels_path);
            end_time = std::chrono::high_resolution_clock::now();
            
            if (load_success) {
                auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                std::cout << "✅ Index loaded in " << load_duration.count() << " μs" << std::endl;
                std::cout << "   Loaded index size: " << loaded_index.size() << " faces" << std::endl;
                
                // Verify loaded index works
                auto verify_result = loaded_index.search_top1(query_feature);
                std::cout << "   Verification search result: " << verify_result.first 
                         << " (score: " << verify_result.second << ")" << std::endl;
                
                if (verify_result.first == names[0]) {
                    std::cout << "   ✅ Loaded index works correctly!" << std::endl;
                } else {
                    std::cout << "   ⚠️  Loaded index results differ" << std::endl;
                }
            } else {
                std::cout << "❌ Failed to load index" << std::endl;
            }
        } else {
            std::cout << "❌ Failed to save index" << std::endl;
        }
        
        std::cout << "\n=== Test 5: Performance Benchmark ===" << std::endl;
        
        // Benchmark search performance
        const int num_queries = 100;
        std::vector<std::vector<float>> query_features;
        for (int i = 0; i < num_queries; ++i) {
            query_features.push_back(generate_random_feature(feature_dim, rng));
        }
        
        start_time = std::chrono::high_resolution_clock::now();
        for (const auto& query : query_features) {
            face_index.search_top1(query);
        }
        end_time = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double avg_time_us = static_cast<double>(total_time.count()) / num_queries;
        double searches_per_second = 1000000.0 / avg_time_us;
        
        std::cout << "Benchmark results (index size: " << face_index.size() << " faces):" << std::endl;
        std::cout << "  Total queries: " << num_queries << std::endl;
        std::cout << "  Total time: " << total_time.count() << " μs" << std::endl;
        std::cout << "  Average time per search: " << avg_time_us << " μs" << std::endl;
        std::cout << "  Searches per second: " << searches_per_second << std::endl;
        
        std::cout << "\n🎉 All FAISS integration tests completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}