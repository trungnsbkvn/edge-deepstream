#include "faiss_index.h"
#include "edge_deepstream.h"
#include "config_parser.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace EdgeDeepStream;

// Generate random normalized feature vector (simulating face embeddings)
std::vector<float> generate_random_feature(int dim, std::mt19937 &rng)
{
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> feature(dim);

    for (int i = 0; i < dim; ++i)
    {
        feature[i] = dist(rng);
    }

    // Normalize for cosine similarity
    float norm = 0.0f;
    for (float v : feature)
    {
        norm += v * v;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-12f)
    {
        for (float &v : feature)
        {
            v /= norm;
        }
    }

    return feature;
}

int main()
{
    std::cout << "=== EdgeDeepStream FAISS Integration Test ===" << std::endl;

#ifdef HAVE_FAISS
    std::cout << "✅ FAISS support enabled" << std::endl;
#else
    std::cout << "❌ FAISS support not available" << std::endl;
    return 1;
#endif

    try
    {
        // Load configuration from TOML
        std::string config_path = "/home/m2n/edge-deepstream/config/config_pipeline.toml";
        auto config = EdgeDeepStream::ConfigParser::parse_toml(config_path);
        if (!config)
        {
            std::cerr << "❌ Failed to parse config file: " << config_path << std::endl;
            return 1;
        }
        std::cout << "✅ Config loaded from " << config_path << std::endl;

        // Extract recognition configuration
        std::string metric = config->get<std::string>("recognition", "metric", "cosine");
        std::string index_type = config->get<std::string>("recognition", "index_type", "flat");
        bool use_gpu = config->get<int>("recognition", "use_gpu", 0) != 0;
        int gpu_id = config->get<int>("recognition", "gpu_id", 0);
        int nlist = config->get<int>("recognition", "nlist", 100);

        std::string index_path = config->get<std::string>("recognition", "index_path", "/home/m2n/edge-deepstream/data/index/faiss.index");
        std::string labels_path = config->get<std::string>("recognition", "labels_path", "/home/m2n/edge-deepstream/data/index/labels.json");

        std::cout << "Recognition config:" << std::endl;
        std::cout << "  metric: " << metric << std::endl;
        std::cout << "  index_type: " << index_type << std::endl;
        std::cout << "  use_gpu: " << (use_gpu ? "true" : "false") << std::endl;
        std::cout << "  gpu_id: " << gpu_id << std::endl;
        std::cout << "  nlist: " << nlist << std::endl;
        std::cout << "  index_path: " << index_path << std::endl;
        std::cout << "  labels_path: " << labels_path << std::endl;

        // Test configuration
        const int feature_dim = 512; // ArcFace standard dimension

        // Create face index with config from TOML
        FaceIndex::Config faiss_config(metric, index_type);
        faiss_config.use_gpu = use_gpu;
        faiss_config.nlist = nlist;
        FaceIndex face_index(feature_dim, faiss_config);

        std::cout << "\n=== Test 1: Loading Existing Index ===" << std::endl;

        // Load existing index and labels
        auto start_time = std::chrono::high_resolution_clock::now();
        bool load_success = face_index.load(index_path, labels_path);
        auto end_time = std::chrono::high_resolution_clock::now();

        if (load_success)
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "✅ Index loaded in " << duration.count() << " μs" << std::endl;
            std::cout << "   Index size: " << face_index.size() << " faces" << std::endl;
            std::cout << "   Labels: ";
            const auto &labels = face_index.get_labels();
            for (size_t i = 0; i < std::min(size_t(5), labels.size()); ++i)
            {
                std::cout << labels[i];
                if (i < std::min(size_t(5), labels.size()) - 1)
                    std::cout << ", ";
            }
            if (labels.size() > 5)
                std::cout << "...";
            std::cout << std::endl;
        }
        else
        {
            std::cout << "❌ Failed to load index from " << index_path << " and " << labels_path << std::endl;
            return 1;
        }

        std::cout << "\n=== Test 2: Face Recognition Search ===" << std::endl;

        // Generate test queries (since we don't have the original features)
        std::mt19937 rng(42); // Fixed seed for reproducible results
        const int num_test_queries = 3;
        std::vector<std::vector<float>> test_queries;

        for (int i = 0; i < num_test_queries; ++i)
        {
            test_queries.push_back(generate_random_feature(feature_dim, rng));
        }

        // Test search with random queries
        const auto &labels = face_index.get_labels();
        for (int i = 0; i < num_test_queries; ++i)
        {
            std::cout << "\nQuery " << (i + 1) << ":" << std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            auto result = face_index.search_top1(test_queries[i]);
            end_time = std::chrono::high_resolution_clock::now();

            auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

            std::cout << "  Result: " << result.first << " (score: " << result.second << ")" << std::endl;
            std::cout << "  Search time: " << search_duration.count() << " μs" << std::endl;
        }

        std::cout << "\n=== Test 3: Top-K Search ===" << std::endl;

        // Test k-nearest neighbors search
        const int k = 3;
        auto query_feature = test_queries[0]; // Use first test query

        start_time = std::chrono::high_resolution_clock::now();
        auto knn_results = face_index.search(query_feature, k);
        end_time = std::chrono::high_resolution_clock::now();

        auto knn_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Top-" << k << " matches for query 1:" << std::endl;
        for (size_t i = 0; i < knn_results.size(); ++i)
        {
            int idx = knn_results[i].first;
            float score = knn_results[i].second;
            if (idx >= 0 && idx < static_cast<int>(labels.size()))
            {
                std::cout << "  " << (i + 1) << ". " << labels[idx] << " (score: " << score << ")" << std::endl;
            }
        }
        std::cout << "Search time: " << knn_duration.count() << " μs" << std::endl;

        std::cout << "\n=== Test 4: Save and Load Index ===" << std::endl;

        // Test saving
        std::string temp_index_path = "/tmp/test_face.index";
        std::string temp_labels_path = "/tmp/test_labels.json";

        start_time = std::chrono::high_resolution_clock::now();
        bool save_success = face_index.save(temp_index_path, temp_labels_path);
        end_time = std::chrono::high_resolution_clock::now();

        if (save_success)
        {
            auto save_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "✅ Index saved in " << save_duration.count() << " μs" << std::endl;

            // Test loading
            FaceIndex loaded_index(feature_dim, faiss_config);

            start_time = std::chrono::high_resolution_clock::now();
            bool load_success = loaded_index.load(temp_index_path, temp_labels_path);
            end_time = std::chrono::high_resolution_clock::now();

            if (load_success)
            {
                auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                std::cout << "✅ Index loaded in " << load_duration.count() << " μs" << std::endl;
                std::cout << "   Loaded index size: " << loaded_index.size() << " faces" << std::endl;

                // Verify loaded index works
                auto verify_result = loaded_index.search_top1(query_feature);
                std::cout << "   Verification search result: " << verify_result.first
                          << " (score: " << verify_result.second << ")" << std::endl;

                // Compare with original search result
                auto original_result = face_index.search_top1(query_feature);
                if (verify_result.first == original_result.first)
                {
                    std::cout << "   ✅ Loaded index works correctly!" << std::endl;
                }
                else
                {
                    std::cout << "   ⚠️  Loaded index results differ" << std::endl;
                }
            }
            else
            {
                std::cout << "❌ Failed to load index" << std::endl;
            }
        }
        else
        {
            std::cout << "❌ Failed to save index" << std::endl;
        }

        std::cout << "\n=== Test 5: Performance Benchmark ===" << std::endl;

        // Benchmark search performance
        const int num_queries = 100;
        std::vector<std::vector<float>> query_features;
        for (int i = 0; i < num_queries; ++i)
        {
            query_features.push_back(generate_random_feature(feature_dim, rng));
        }

        start_time = std::chrono::high_resolution_clock::now();
        for (const auto &query : query_features)
        {
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
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}