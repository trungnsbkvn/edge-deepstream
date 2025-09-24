#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>
#include <opencv2/opencv.hpp>

namespace EdgeDeepStream {

// Forward declarations
class FaceIndex;
class TensorRTInfer;

/**
 * Face enrollment and management operations
 * Handles adding, updating, and deleting persons from the face recognition system
 */
class EnrollOps {
public:
    EnrollOps();
    ~EnrollOps();
    
    // Disable copy, enable move
    EnrollOps(const EnrollOps&) = delete;
    EnrollOps& operator=(const EnrollOps&) = delete;
    EnrollOps(EnrollOps&&) = default;
    EnrollOps& operator=(EnrollOps&&) = default;
    
    /**
     * Initialize enrollment operations with configuration
     */
    bool initialize(const std::string& config_path);
    
    /**
     * Set the FAISS index and TensorRT engine for operations
     */
    void set_components(std::shared_ptr<FaceIndex> index, 
                       std::shared_ptr<TensorRTInfer> engine);
    
    // Face processing operations
    
    /**
     * Detect face bounding box using Haar cascades (fallback method)
     */
    std::optional<cv::Rect> detect_face_bbox(const cv::Mat& bgr_image);
    
    /**
     * Crop and align face to 112x112 with margin
     */
    std::optional<cv::Mat> crop_align_112(const cv::Mat& bgr_image, float margin = 0.10f);
    
    /**
     * Calculate blur variance for quality assessment
     */
    double calculate_blur_variance(const cv::Mat& bgr_image);
    
    /**
     * Extract face embedding using ArcFace model
     */
    std::optional<std::vector<float>> extract_embedding(const cv::Mat& face112_bgr);
    
    // Index and metadata operations
    
    /**
     * Person metadata structure
     */
    struct PersonRecord {
        std::string user_id;
        std::string name;
        std::vector<std::string> aligned_paths;
        int64_t start_time = 0;
        int64_t end_time = 0;
    };
    
    /**
     * Labels metadata structure  
     */
    struct LabelsMetadata {
        int version = 2;
        std::vector<std::string> labels;
        std::unordered_map<std::string, PersonRecord> persons;
    };
    
    /**
     * Load or create FAISS index from configuration
     */
    std::shared_ptr<FaceIndex> load_or_create_index(const std::unordered_map<std::string, std::string>& config);
    
    /**
     * Read labels metadata from JSON file
     */
    LabelsMetadata read_labels(const std::string& labels_path);
    
    /**
     * Write labels metadata to JSON file
     */
    bool write_labels(const std::string& labels_path, const LabelsMetadata& metadata);
    
    /**
     * Add or update person metadata
     */
    void upsert_person_meta(LabelsMetadata& metadata, const std::string& user_id,
                           const std::string& name, const std::string& aligned_path = "");
    
    /**
     * Search result structure
     */
    struct SearchResult {
        std::string label;
        double score = -1.0;
        bool found = false;
    };
    
    /**
     * Search for top match in index
     */
    SearchResult search_top(const std::vector<float>& embedding);
    
    // High-level enrollment operations
    
    /**
     * Enrollment result structure
     */
    struct EnrollmentResult {
        bool success = false;
        std::string message;
        std::string user_id;
        double blur_score = 0.0;
        double similarity_score = -1.0;
        int vectors_added = 0;
    };
    
    /**
     * Enroll a person from image data
     */
    EnrollmentResult enroll_person(const std::string& user_id, const std::string& name,
                                  const cv::Mat& image_data, 
                                  double blur_threshold = 100.0,
                                  double similarity_threshold = 0.7);
    
    /**
     * Enroll a person from image file
     */
    EnrollmentResult enroll_person_from_file(const std::string& user_id, const std::string& name,
                                            const std::string& image_path,
                                            double blur_threshold = 100.0,
                                            double similarity_threshold = 0.7);
    
    /**
     * Delete result structure
     */
    struct DeletionResult {
        bool success = false;
        std::string message;
        int vectors_removed = 0;
        int files_removed = 0;
    };
    
    /**
     * Delete a person from the system
     */
    DeletionResult delete_person(const std::string& user_id, bool remove_files = true);
    
    /**
     * Delete multiple persons (e.g., "user1,user2" or "all")
     */
    DeletionResult delete_persons(const std::string& target_list, bool remove_files = true);
    
    /**
     * List all enrolled persons
     */
    std::vector<PersonRecord> list_persons();
    
    /**
     * Get person information
     */
    std::optional<PersonRecord> get_person(const std::string& user_id);
    
    /**
     * Get enrollment statistics
     */
    struct EnrollmentStats {
        int total_persons = 0;
        int total_vectors = 0;
        std::string index_type;
        std::string metric;
        bool gpu_enabled = false;
    };
    
    EnrollmentStats get_stats();
    
private:
    std::shared_ptr<FaceIndex> faiss_index_;
    std::shared_ptr<TensorRTInfer> tensorrt_engine_;
    
    // Configuration
    std::string config_path_;
    std::string index_path_;
    std::string labels_path_;
    std::string faces_data_path_;
    std::string aligned_dir_;
    std::string engine_path_;
    
    // Face detection cascade
    cv::CascadeClassifier haar_cascade_;
    bool haar_initialized_ = false;
    
    /**
     * Initialize Haar cascade for face detection
     */
    bool initialize_haar_cascade();
    
    /**
     * Resolve engine path from secondary-gie config file (matching Python logic)
     */
    std::string resolve_engine_path_from_config(const std::string& sgie_config_path);
    
    /**
     * Fallback engine path search
     */
    std::string fallback_engine_search();
    
    /**
     * Save aligned face image
     */
    std::string save_aligned_face(const cv::Mat& face112, const std::string& user_id);
    
    /**
     * Generate unique filename for aligned face
     */
    std::string generate_aligned_filename(const std::string& user_id);
    
    /**
     * Remove file safely
     */
    bool remove_file(const std::string& file_path);
};

} // namespace EdgeDeepStream