#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#ifdef HAVE_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#endif

namespace EdgeDeepStream {

/**
 * C++ wrapper for FAISS face feature indexing
 * 
 * This class provides similar functionality to the Python FaceIndex:
 * - Load/save FAISS indices and label mappings
 * - Add face features with associated labels
 * - Search for similar faces using cosine similarity
 * - Support for both flat and IVF index types
 */
class FaceIndex {
public:
    struct Config {
        std::string metric;      // cosine or l2
        std::string index_type;  // flat or ivf
        bool use_gpu;            // GPU support (if available)
        int nlist;               // For IVF indices
        
        Config() : metric("cosine"), index_type("flat"), use_gpu(false), nlist(100) {}
        Config(const std::string& m, const std::string& t) 
            : metric(m), index_type(t), use_gpu(false), nlist(100) {}
    };

    // Constructor
    explicit FaceIndex(int dimension = 512, const Config& config = Config{});
    
    // Destructor
    ~FaceIndex();
    
    // Copy/move operations
    FaceIndex(const FaceIndex&) = delete;
    FaceIndex& operator=(const FaceIndex&) = delete;
    FaceIndex(FaceIndex&&) = default;
    FaceIndex& operator=(FaceIndex&&) = default;

    // Loading and saving
    bool load(const std::string& index_path, const std::string& labels_path);
    bool save(const std::string& index_path, const std::string& labels_path) const;
    
    // Building from directory of .npy files (similar to Python from_dir)
    static std::unique_ptr<FaceIndex> from_dir(const std::string& dir_path, const Config& config = Config{});
    
    // Adding features
    bool add(const std::vector<std::string>& names, const std::vector<std::vector<float>>& vectors);
    bool add_single(const std::string& name, const std::vector<float>& vector);
    
    // Removing features
    int remove_label(const std::string& label);
    
    // Searching
    std::vector<std::pair<int, float>> search(const std::vector<float>& query_vector, int k = 1) const;
    std::pair<std::string, float> search_top1(const std::vector<float>& query_vector) const;
    
    // Batch search
    std::vector<std::vector<std::pair<std::string, float>>> search_batch(
        const std::vector<std::vector<float>>& query_vectors, int k = 1) const;
    
    // Information
    int size() const { return static_cast<int>(labels_.size()); }
    int dimension() const { return dim_; }
    bool is_trained() const;
    bool empty() const { return size() == 0; }
    
    // Get labels
    const std::vector<std::string>& get_labels() const { return labels_; }
    
    // Configuration
    const Config& get_config() const { return config_; }

private:
    void normalize_vector(std::vector<float>& vector) const;
    std::vector<float> normalize_vector_copy(const std::vector<float>& vector) const;
    bool create_index();
    bool train_if_needed();
    std::vector<std::vector<float>> reconstruct_all() const;

#ifdef HAVE_FAISS
    std::unique_ptr<faiss::Index> index_;
#else
    void* index_;  // Placeholder when FAISS not available
#endif
    
    std::vector<std::string> labels_;  // label for each vector in the index
    Config config_;
    int dim_;
    bool trained_;
    mutable bool search_needs_normalization_;  // Cache normalization requirement
};

} // namespace EdgeDeepStream