#include "faiss_index.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <sstream>

#ifdef HAVE_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/MetricType.h>
#include <faiss/utils/utils.h>
#endif

// JSON parsing for labels (simple implementation)
#include <map>

namespace EdgeDeepStream {

namespace {
    // Simple JSON parser for labels file
    std::map<std::string, std::vector<std::string>> parse_labels_json(const std::string& filename) {
        std::map<std::string, std::vector<std::string>> result;
        std::ifstream file(filename);
        if (!file.is_open()) {
            return result;
        }
        
        std::string content;
        std::string line;
        while (std::getline(file, line)) {
            content += line;
        }
        
        // Simple JSON parsing - look for "labels": ["label1", "label2", ...]
        size_t labels_pos = content.find("\"labels\"");
        if (labels_pos != std::string::npos) {
            size_t bracket_start = content.find("[", labels_pos);
            size_t bracket_end = content.find("]", bracket_start);
            if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                std::string labels_str = content.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                
                // Parse individual labels
                std::vector<std::string> labels;
                std::stringstream ss(labels_str);
                std::string label;
                
                while (std::getline(ss, label, ',')) {
                    // Remove quotes and whitespace
                    label.erase(0, label.find_first_not_of(" \t\"\n\r"));
                    label.erase(label.find_last_not_of(" \t\"\n\r") + 1);
                    if (!label.empty()) {
                        labels.push_back(label);
                    }
                }
                result["labels"] = labels;
            }
        }
        return result;
    }
    
    void save_labels_json(const std::string& filename, const std::vector<std::string>& labels) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open labels file for writing: " + filename);
        }
        
        file << "{\n  \"labels\": [";
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i > 0) file << ",";
            file << "\n    \"" << labels[i] << "\"";
        }
        file << "\n  ]\n}\n";
    }
}

FaceIndex::FaceIndex(int dimension, const Config& config)
    : config_(config), dim_(dimension), trained_(false), search_needs_normalization_(config.metric == "cosine") {
    
#ifdef HAVE_FAISS
    index_ = nullptr;
#else
    index_ = nullptr;
    std::cerr << "Warning: FAISS not available. FaceIndex will not work properly." << std::endl;
#endif
}

FaceIndex::~FaceIndex() = default;

bool FaceIndex::create_index() {
#ifdef HAVE_FAISS
    try {
        faiss::MetricType metric_type = faiss::METRIC_L2;
        if (config_.metric == "cosine") {
            metric_type = faiss::METRIC_INNER_PRODUCT;
        }
        
        if (config_.index_type == "flat") {
            index_ = std::make_unique<faiss::IndexFlat>(dim_, metric_type);
            trained_ = true;  // Flat index doesn't need training
        } else if (config_.index_type == "ivf") {
            auto quantizer = std::make_unique<faiss::IndexFlat>(dim_, metric_type);
            auto ivf_index = std::make_unique<faiss::IndexIVFFlat>(quantizer.get(), dim_, config_.nlist, metric_type);
            quantizer.release();  // IndexIVFFlat takes ownership
            index_ = std::move(ivf_index);
            trained_ = false;  // IVF index needs training
        } else {
            std::cerr << "Unsupported index type: " << config_.index_type << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error creating FAISS index: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "FAISS not available" << std::endl;
    return false;
#endif
}

void FaceIndex::normalize_vector(std::vector<float>& vector) const {
    if (config_.metric != "cosine") return;
    
    float norm = 0.0f;
    for (float v : vector) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-12f) {
        for (float& v : vector) {
            v /= norm;
        }
    }
}

std::vector<float> FaceIndex::normalize_vector_copy(const std::vector<float>& vector) const {
    std::vector<float> result = vector;
    normalize_vector(result);
    return result;
}

bool FaceIndex::train_if_needed() {
#ifdef HAVE_FAISS
    if (trained_ || !index_) return true;
    
    try {
        // For IVF indices, we need training data
        auto* ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(index_.get());
        if (ivf_index) {
            // We need at least nlist vectors for training
            if (labels_.size() < static_cast<size_t>(config_.nlist)) {
                std::cerr << "Not enough vectors for IVF training: " << labels_.size() 
                         << " < " << config_.nlist << std::endl;
                return false;
            }
            
            // Use existing vectors for training (this is a simplification)
            // In a real implementation, you might want separate training data
            ivf_index->train(labels_.size(), reinterpret_cast<const float*>(labels_.data()));
        }
        
        trained_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error training FAISS index: " << e.what() << std::endl;
        return false;
    }
#else
    return false;
#endif
}

bool FaceIndex::load(const std::string& index_path, const std::string& labels_path) {
#ifdef HAVE_FAISS
    try {
        // Load the FAISS index
        index_.reset(faiss::read_index(index_path.c_str()));
        if (!index_) {
            std::cerr << "Failed to load FAISS index from: " << index_path << std::endl;
            return false;
        }
        
        dim_ = index_->d;
        trained_ = index_->is_trained;
        
        // Load labels
        auto labels_data = parse_labels_json(labels_path);
        if (labels_data.find("labels") != labels_data.end()) {
            labels_ = labels_data["labels"];
        } else {
            std::cerr << "Warning: No labels found in " << labels_path << std::endl;
            labels_.clear();
        }
        
        std::cout << "Loaded FAISS index with " << index_->ntotal << " vectors and " 
                  << labels_.size() << " labels" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading FAISS index: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "FAISS not available" << std::endl;
    return false;
#endif
}

bool FaceIndex::save(const std::string& index_path, const std::string& labels_path) const {
#ifdef HAVE_FAISS
    try {
        if (!index_) {
            std::cerr << "No index to save" << std::endl;
            return false;
        }
        
        // Create directories if needed
        std::filesystem::create_directories(std::filesystem::path(index_path).parent_path());
        std::filesystem::create_directories(std::filesystem::path(labels_path).parent_path());
        
        // Save FAISS index
        faiss::write_index(index_.get(), index_path.c_str());
        
        // Save labels
        save_labels_json(labels_path, labels_);
        
        std::cout << "Saved FAISS index with " << index_->ntotal << " vectors to " << index_path << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving FAISS index: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "FAISS not available" << std::endl;
    return false;
#endif
}

bool FaceIndex::add_single(const std::string& name, const std::vector<float>& vector) {
    return add({name}, {vector});
}

bool FaceIndex::add(const std::vector<std::string>& names, const std::vector<std::vector<float>>& vectors) {
#ifdef HAVE_FAISS
    if (names.size() != vectors.size()) {
        std::cerr << "Names and vectors size mismatch" << std::endl;
        return false;
    }
    
    if (vectors.empty()) {
        return true;  // Nothing to add
    }
    
    // Validate dimensions
    for (const auto& vec : vectors) {
        if (static_cast<int>(vec.size()) != dim_) {
            std::cerr << "Vector dimension mismatch: expected " << dim_ << ", got " << vec.size() << std::endl;
            return false;
        }
    }
    
    try {
        // Create index if it doesn't exist
        if (!index_ && !create_index()) {
            return false;
        }
        
        // Prepare normalized vectors if needed
        std::vector<float> flat_vectors;
        flat_vectors.reserve(vectors.size() * dim_);
        
        for (const auto& vec : vectors) {
            if (config_.metric == "cosine") {
                auto normalized = normalize_vector_copy(vec);
                flat_vectors.insert(flat_vectors.end(), normalized.begin(), normalized.end());
            } else {
                flat_vectors.insert(flat_vectors.end(), vec.begin(), vec.end());
            }
        }
        
        // Train if needed (for IVF indices)
        if (!trained_) {
            if (!train_if_needed()) {
                return false;
            }
        }
        
        // Add vectors to index
        index_->add(vectors.size(), flat_vectors.data());
        
        // Add labels
        labels_.insert(labels_.end(), names.begin(), names.end());
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error adding vectors to FAISS index: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "FAISS not available" << std::endl;
    return false;
#endif
}

std::vector<std::pair<int, float>> FaceIndex::search(const std::vector<float>& query_vector, int k) const {
#ifdef HAVE_FAISS
    if (!index_) {
        std::cerr << "Index not initialized" << std::endl;
        return {};
    }
    
    if (static_cast<int>(query_vector.size()) != dim_) {
        std::cerr << "Query vector dimension mismatch" << std::endl;
        return {};
    }
    
    try {
        // Normalize query vector if needed
        std::vector<float> query = config_.metric == "cosine" ? 
            normalize_vector_copy(query_vector) : query_vector;
        
        // Perform search
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distances(k);
        
        index_->search(1, query.data(), k, distances.data(), indices.data());
        
        // Convert results
        std::vector<std::pair<int, float>> results;
        results.reserve(k);
        
        for (int i = 0; i < k; ++i) {
            if (indices[i] >= 0 && indices[i] < static_cast<faiss::idx_t>(labels_.size())) {
                results.emplace_back(static_cast<int>(indices[i]), distances[i]);
            }
        }
        
        return results;
    } catch (const std::exception& e) {
        std::cerr << "Error in FAISS search: " << e.what() << std::endl;
        return {};
    }
#else
    std::cerr << "FAISS not available" << std::endl;
    return {};
#endif
}

std::pair<std::string, float> FaceIndex::search_top1(const std::vector<float>& query_vector) const {
    auto results = search(query_vector, 1);
    if (!results.empty() && results[0].first < static_cast<int>(labels_.size())) {
        return {labels_[results[0].first], results[0].second};
    }
    return {"", -1.0f};
}

std::vector<std::vector<std::pair<std::string, float>>> FaceIndex::search_batch(
    const std::vector<std::vector<float>>& query_vectors, int k) const {
    
    std::vector<std::vector<std::pair<std::string, float>>> batch_results;
    batch_results.reserve(query_vectors.size());
    
    for (const auto& query : query_vectors) {
        auto results = search(query, k);
        std::vector<std::pair<std::string, float>> named_results;
        named_results.reserve(results.size());
        
        for (const auto& [idx, score] : results) {
            if (idx >= 0 && idx < static_cast<int>(labels_.size())) {
                named_results.emplace_back(labels_[idx], score);
            }
        }
        batch_results.push_back(std::move(named_results));
    }
    
    return batch_results;
}

bool FaceIndex::is_trained() const {
#ifdef HAVE_FAISS
    return trained_ && index_ && index_->is_trained;
#else
    return false;
#endif
}

std::unique_ptr<FaceIndex> FaceIndex::from_dir(const std::string& dir_path, const Config& config) {
    // This would implement loading .npy files from directory
    // For now, return a basic implementation
    auto face_index = std::make_unique<FaceIndex>(512, config);
    
    std::cout << "FaceIndex::from_dir not fully implemented yet for: " << dir_path << std::endl;
    std::cout << "Please use load() method to load existing FAISS indices." << std::endl;
    
    return face_index;
}

} // namespace EdgeDeepStream