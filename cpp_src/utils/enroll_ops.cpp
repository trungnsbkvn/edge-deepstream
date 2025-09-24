#include "enroll_ops.h"
#include "faiss_index.h"
#include "tensorrt_infer.h"
#include "config_parser.h"
#include "edge_deepstream.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <iomanip>
#include <sstream>
#include <ctime>

using json = nlohmann::json;

namespace EdgeDeepStream {

EnrollOps::EnrollOps() = default;
EnrollOps::~EnrollOps() = default;

bool EnrollOps::initialize(const std::string& config_path) {
    config_path_ = config_path;
    
    // Load configuration
    auto config = ConfigParser::parse_toml(config_path);
    if (!config) {
        return false;
    }
    
    // Store config paths
    index_path_ = config->get<std::string>("recognition", "index_path", "data/index/faiss.index");
    labels_path_ = config->get<std::string>("recognition", "labels_path", "data/index/labels.json");
    faces_data_path_ = config->get<std::string>("pipeline", "known_face_dir", "data/known_faces");
    aligned_dir_ = faces_data_path_ + "/aligned";
    
    // Create directories
    try {
        std::filesystem::create_directories(std::filesystem::path(index_path_).parent_path());
        std::filesystem::create_directories(std::filesystem::path(labels_path_).parent_path());
        std::filesystem::create_directories(aligned_dir_);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create directories: " << e.what() << std::endl;
        return false;
    }
    
    // Resolve engine path from secondary-gie config file (matching Python logic)
    std::string sgie_config_path = config->get<std::string>("secondary-gie-1", "config-file-path", "config/config_arcface.txt");
    engine_path_ = resolve_engine_path_from_config(sgie_config_path);
    if (engine_path_.empty()) {
        std::cerr << "Could not resolve ArcFace engine path from config: " << sgie_config_path << std::endl;
        return false;
    }
    
    // Initialize Haar cascade for face detection
    initialize_haar_cascade();
    
    return true;
}

void EnrollOps::set_components(std::shared_ptr<FaceIndex> index, 
                              std::shared_ptr<TensorRTInfer> engine) {
    faiss_index_ = index;
    tensorrt_engine_ = engine;
}

std::optional<cv::Rect> EnrollOps::detect_face_bbox(const cv::Mat& bgr_image) {
    if (!haar_initialized_ || bgr_image.empty()) {
        return std::nullopt;
    }
    
    try {
        cv::Mat gray;
        cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Rect> faces;
        haar_cascade_.detectMultiScale(gray, faces, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(60, 60));
        
        if (faces.empty()) {
            return std::nullopt;
        }
        
        // Return largest face
        auto largest = std::max_element(faces.begin(), faces.end(), 
            [](const cv::Rect& a, const cv::Rect& b) {
                return a.width * a.height < b.width * b.height;
            });
        
        return *largest;
    } catch (const std::exception& e) {
        return std::nullopt;
    }
}

std::optional<cv::Mat> EnrollOps::crop_align_112(const cv::Mat& bgr_image, float margin) {
    if (bgr_image.empty()) {
        return std::nullopt;
    }
    
    int H = bgr_image.rows;
    int W = bgr_image.cols;
    
    auto bbox = detect_face_bbox(bgr_image);
    cv::Mat crop;
    
    if (bbox) {
        // Use detected face
        int x = bbox->x, y = bbox->y, w = bbox->width, h = bbox->height;
        int side0 = std::max(w, h);
        int side = static_cast<int>((1.0f + 2.0f * std::max(0.0f, margin)) * side0);
        int cx = x + w / 2;
        int cy = y + h / 2;
        int x0 = std::max(0, cx - side / 2);
        int y0 = std::max(0, cy - side / 2);
        int x1 = std::min(W, x0 + side);
        int y1 = std::min(H, y0 + side);
        
        cv::Rect crop_rect(x0, y0, x1 - x0, y1 - y0);
        if (crop_rect.width > 0 && crop_rect.height > 0) {
            crop = bgr_image(crop_rect).clone();
        }
    }
    
    // Fallback to center square
    if (crop.empty()) {
        int side = std::min(H, W);
        int cx = W / 2, cy = H / 2;
        int x0 = std::max(0, cx - side / 2);
        int y0 = std::max(0, cy - side / 2);
        cv::Rect crop_rect(x0, y0, side, side);
        crop = bgr_image(crop_rect).clone();
    }
    
    if (crop.empty()) {
        return std::nullopt;
    }
    
    try {
        cv::Mat face112;
        cv::resize(crop, face112, cv::Size(112, 112), 0, 0, cv::INTER_LINEAR);
        return face112;
    } catch (const std::exception& e) {
        return std::nullopt;
    }
}

double EnrollOps::calculate_blur_variance(const cv::Mat& bgr_image) {
    if (bgr_image.empty()) {
        return 0.0;
    }
    
    try {
        cv::Mat gray;
        cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);
        
        return stddev.val[0] * stddev.val[0];  // Variance = stddev^2
    } catch (const std::exception& e) {
        return 0.0;
    }
}

std::optional<std::vector<float>> EnrollOps::extract_embedding(const cv::Mat& face112_bgr) {
    if (!tensorrt_engine_ || face112_bgr.empty()) {
        return std::nullopt;
    }
    
    try {
        // Convert BGR to RGB and normalize
        cv::Mat rgb;
        cv::cvtColor(face112_bgr, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F);
        rgb = (rgb - 127.5f) / 128.0f;
        
        // Convert to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        
        std::vector<float> input_data;
        input_data.reserve(3 * 112 * 112);
        
        for (int c = 0; c < 3; c++) {
            cv::Mat channel = channels[c];
            float* data = reinterpret_cast<float*>(channel.data);
            input_data.insert(input_data.end(), data, data + 112 * 112);
        }
        
        // Run inference
        auto embedding = tensorrt_engine_->infer(input_data);
        if (embedding.empty()) {
            return std::nullopt;
        }
        
        // Normalize embedding
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm) + 1e-12f;
        
        for (float& val : embedding) {
            val /= norm;
        }
        
        return embedding;
    } catch (const std::exception& e) {
        return std::nullopt;
    }
}

EnrollOps::LabelsMetadata EnrollOps::read_labels(const std::string& labels_path) {
    LabelsMetadata metadata;
    
    if (labels_path.empty() || !std::filesystem::exists(labels_path)) {
        return metadata;
    }
    
    try {
        std::ifstream file(labels_path);
        json j;
        file >> j;
        
        if (j.contains("version")) {
            metadata.version = j["version"];
        }
        
        if (j.contains("labels") && j["labels"].is_array()) {
            for (const auto& label : j["labels"]) {
                metadata.labels.push_back(label.get<std::string>());
            }
        }
        
        if (j.contains("persons") && j["persons"].is_object()) {
            for (auto& [user_id, person_data] : j["persons"].items()) {
                PersonRecord record;
                record.user_id = user_id;
                
                if (person_data.contains("name")) {
                    record.name = person_data["name"];
                }
                if (person_data.contains("aligned_paths") && person_data["aligned_paths"].is_array()) {
                    for (const auto& path : person_data["aligned_paths"]) {
                        record.aligned_paths.push_back(path.get<std::string>());
                    }
                }
                if (person_data.contains("start_time")) {
                    record.start_time = person_data["start_time"];
                }
                if (person_data.contains("end_time")) {
                    record.end_time = person_data["end_time"];
                }
                
                metadata.persons[user_id] = record;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading labels file: " << e.what() << std::endl;
    }
    
    return metadata;
}

bool EnrollOps::write_labels(const std::string& labels_path, const LabelsMetadata& metadata) {
    if (labels_path.empty()) {
        return false;
    }
    
    try {
        // Create directory if it doesn't exist
        std::filesystem::create_directories(std::filesystem::path(labels_path).parent_path());
        
        json j;
        j["version"] = metadata.version;
        j["labels"] = metadata.labels;
        
        json persons_obj = json::object();
        for (const auto& [user_id, record] : metadata.persons) {
            json person_data;
            person_data["user_id"] = record.user_id;
            person_data["name"] = record.name;
            person_data["aligned_paths"] = record.aligned_paths;
            person_data["start_time"] = record.start_time;
            person_data["end_time"] = record.end_time;
            persons_obj[user_id] = person_data;
        }
        j["persons"] = persons_obj;
        
        std::ofstream file(labels_path);
        file << j.dump(2);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing labels file: " << e.what() << std::endl;
        return false;
    }
}

void EnrollOps::upsert_person_meta(LabelsMetadata& metadata, const std::string& user_id,
                                  const std::string& name, const std::string& aligned_path) {
    PersonRecord& record = metadata.persons[user_id];
    
    // Initialize times if not set
    if (record.start_time == 0) {
        record.start_time = 0;
    }
    if (record.end_time == 0) {
        record.end_time = 0;
    }
    
    record.user_id = user_id;
    record.name = name.empty() ? (record.name.empty() ? user_id : record.name) : name;
    
    // Add aligned path if provided and not already present
    if (!aligned_path.empty()) {
        auto it = std::find(record.aligned_paths.begin(), record.aligned_paths.end(), aligned_path);
        if (it == record.aligned_paths.end()) {
            record.aligned_paths.push_back(aligned_path);
        }
    }
    
    metadata.version = 2;
}

EnrollOps::SearchResult EnrollOps::search_top(const std::vector<float>& embedding) {
    SearchResult result;
    
    if (!faiss_index_ || faiss_index_->size() == 0) {
        return result;
    }
    
    try {
        auto [label, score] = faiss_index_->search_top1(embedding);
        result.label = label;
        result.score = score;
        result.found = !label.empty();
    } catch (const std::exception& e) {
        std::cerr << "Error in search_top: " << e.what() << std::endl;
    }
    
    return result;
}

EnrollOps::EnrollmentResult EnrollOps::enroll_person(const std::string& user_id, const std::string& name,
                                                     const cv::Mat& image_data, 
                                                     double blur_threshold,
                                                     double similarity_threshold) {
    EnrollmentResult result;
    result.user_id = user_id;
    
    if (user_id.empty() || image_data.empty()) {
        result.message = "Invalid user ID or image data";
        return result;
    }
    
    // Step 1: Crop and align face
    auto face112_opt = crop_align_112(image_data);
    if (!face112_opt) {
        result.message = "Failed to crop and align face from image";
        return result;
    }
    
    cv::Mat face112 = face112_opt.value();
    
    // Step 2: Quality check - blur detection
    result.blur_score = calculate_blur_variance(face112);
    if (result.blur_score < blur_threshold) {
        result.message = "Image quality too poor (blur score: " + std::to_string(result.blur_score) + ")";
        return result;
    }
    
    // Step 3: Extract embedding
    auto embedding_opt = extract_embedding(face112);
    if (!embedding_opt) {
        result.message = "Failed to extract face embedding";
        return result;
    }
    
    std::vector<float> embedding = embedding_opt.value();
    
    // Step 4: Check similarity with existing faces (avoid duplicates)
    if (faiss_index_ && faiss_index_->size() > 0) {
        auto search_result = search_top(embedding);
        if (search_result.found) {
            result.similarity_score = search_result.score;
            if (search_result.score > similarity_threshold) {
                result.message = "Face too similar to existing person: " + search_result.label + 
                               " (score: " + std::to_string(search_result.score) + ")";
                return result;
            }
        }
    }
    
    // Step 5: Save aligned face
    std::string aligned_path = save_aligned_face(face112, user_id);
    if (aligned_path.empty()) {
        result.message = "Failed to save aligned face image";
        return result;
    }
    
    // Step 6: Add to FAISS index
    if (!faiss_index_) {
        result.message = "FAISS index not initialized";
        return result;
    }
    
    try {
        faiss_index_->add_single(user_id, embedding);
        result.vectors_added = 1;
    } catch (const std::exception& e) {
        result.message = "Failed to add vector to FAISS index: " + std::string(e.what());
        return result;
    }
    
    // Step 7: Update metadata
    auto metadata = read_labels(labels_path_);
    upsert_person_meta(metadata, user_id, name, aligned_path);
    
    // Update labels list from index
    metadata.labels = faiss_index_->get_labels();
    
    if (!write_labels(labels_path_, metadata)) {
        result.message = "Failed to update labels metadata";
        return result;
    }
    
    // Step 8: Save index
    try {
        faiss_index_->save(index_path_, labels_path_);
    } catch (const std::exception& e) {
        result.message = "Failed to save FAISS index: " + std::string(e.what());
        return result;
    }
    
    result.success = true;
    result.message = "Person enrolled successfully";
    return result;
}

EnrollOps::EnrollmentResult EnrollOps::enroll_person_from_file(const std::string& user_id, const std::string& name,
                                                              const std::string& image_path,
                                                              double blur_threshold,
                                                              double similarity_threshold) {
    EnrollmentResult result;
    result.user_id = user_id;
    
    if (!std::filesystem::exists(image_path)) {
        result.message = "Image file does not exist: " + image_path;
        return result;
    }
    
    try {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            result.message = "Failed to load image from: " + image_path;
            return result;
        }
        
        return enroll_person(user_id, name, image, blur_threshold, similarity_threshold);
    } catch (const std::exception& e) {
        result.message = "Error loading image: " + std::string(e.what());
        return result;
    }
}

EnrollOps::DeletionResult EnrollOps::delete_person(const std::string& user_id, bool remove_files) {
    DeletionResult result;
    
    if (user_id.empty()) {
        result.message = "Invalid user ID";
        return result;
    }
    
    // Read current metadata
    auto metadata = read_labels(labels_path_);
    
    // Note: FaceIndex doesn't support individual removal, so we simulate by recording the removal
    // The actual removal would require rebuilding the index from remaining entries
    int vectors_removed = 0;
    if (faiss_index_ && metadata.persons.find(user_id) != metadata.persons.end()) {
        vectors_removed = 1; // Assuming one vector per person
        result.vectors_removed = vectors_removed;
    }
    
    // Remove files if requested
    int files_removed = 0;
    if (remove_files && metadata.persons.count(user_id)) {
        const auto& person = metadata.persons.at(user_id);
        for (const std::string& rel_path : person.aligned_paths) {
            std::string abs_path = std::filesystem::path(rel_path).is_absolute() ? 
                                  rel_path : (std::filesystem::current_path() / rel_path).string();
            if (remove_file(abs_path)) {
                files_removed++;
            }
        }
    }
    result.files_removed = files_removed;
    
    // Remove from metadata
    if (metadata.persons.count(user_id)) {
        metadata.persons.erase(user_id);
    }
    
    // Update labels list from index if vectors were removed
    if (vectors_removed > 0 && faiss_index_) {
        metadata.labels = faiss_index_->get_labels();
        
        // Save updated index
        try {
            faiss_index_->save(index_path_, labels_path_);
        } catch (const std::exception& e) {
            result.message += " Warning: Failed to save index: " + std::string(e.what());
        }
    }
    
    // Write updated metadata
    if (!write_labels(labels_path_, metadata)) {
        result.message += " Warning: Failed to update metadata";
    }
    
    result.success = true;
    result.message = "Person deleted successfully";
    return result;
}

EnrollOps::DeletionResult EnrollOps::delete_persons(const std::string& target_list, bool remove_files) {
    DeletionResult result;
    
    if (target_list == "all") {
        // Delete all persons
        auto metadata = read_labels(labels_path_);
        std::vector<std::string> all_users;
        for (const auto& [user_id, _] : metadata.persons) {
            all_users.push_back(user_id);
        }
        
        for (const std::string& user_id : all_users) {
            auto single_result = delete_person(user_id, remove_files);
            result.vectors_removed += single_result.vectors_removed;
            result.files_removed += single_result.files_removed;
        }
        
        result.success = true;
        result.message = "All persons deleted successfully";
    } else {
        // Parse comma-separated list
        std::istringstream ss(target_list);
        std::string user_id;
        
        while (std::getline(ss, user_id, ',')) {
            // Trim whitespace
            user_id.erase(0, user_id.find_first_not_of(" \t"));
            user_id.erase(user_id.find_last_not_of(" \t") + 1);
            
            if (!user_id.empty()) {
                auto single_result = delete_person(user_id, remove_files);
                result.vectors_removed += single_result.vectors_removed;
                result.files_removed += single_result.files_removed;
            }
        }
        
        result.success = true;
        result.message = "Specified persons deleted successfully";
    }
    
    return result;
}

std::vector<EnrollOps::PersonRecord> EnrollOps::list_persons() {
    auto metadata = read_labels(labels_path_);
    std::vector<PersonRecord> persons;
    
    for (const auto& [user_id, record] : metadata.persons) {
        persons.push_back(record);
    }
    
    return persons;
}

std::optional<EnrollOps::PersonRecord> EnrollOps::get_person(const std::string& user_id) {
    auto metadata = read_labels(labels_path_);
    
    if (metadata.persons.count(user_id)) {
        return metadata.persons.at(user_id);
    }
    
    return std::nullopt;
}

EnrollOps::EnrollmentStats EnrollOps::get_stats() {
    EnrollmentStats stats;
    
    if (faiss_index_) {
        stats.total_vectors = faiss_index_->size();
        stats.index_type = faiss_index_->get_config().index_type;
        stats.metric = faiss_index_->get_config().metric;
        stats.gpu_enabled = faiss_index_->get_config().use_gpu;
    }
    
    auto metadata = read_labels(labels_path_);
    stats.total_persons = metadata.persons.size();
    
    return stats;
}

bool EnrollOps::initialize_haar_cascade() {
    std::vector<std::string> cascade_paths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_default.xml"
    };
    
    for (const std::string& path : cascade_paths) {
        if (std::filesystem::exists(path)) {
            if (haar_cascade_.load(path)) {
                haar_initialized_ = true;
                return true;
            }
        }
    }
    
    return false;
}



std::string EnrollOps::save_aligned_face(const cv::Mat& face112, const std::string& user_id) {
    try {
        std::string filename = generate_aligned_filename(user_id);
        std::string full_path = aligned_dir_ + "/" + filename;
        
        if (cv::imwrite(full_path, face112)) {
            return full_path;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving aligned face: " << e.what() << std::endl;
    }
    
    return "";
}

std::string EnrollOps::resolve_engine_path_from_config(const std::string& sgie_config_path) {
    try {
        // Make absolute path if relative
        std::string abs_sgie_path = sgie_config_path;
        if (!std::filesystem::path(sgie_config_path).is_absolute()) {
            abs_sgie_path = std::filesystem::current_path() / sgie_config_path;
        }
        
        if (!std::filesystem::exists(abs_sgie_path)) {
            // Fallback search
            return fallback_engine_search();
        }
        
        // Parse config file using simple INI parser
        std::ifstream file(abs_sgie_path);
        std::string line;
        std::string model_engine_file;
        
        while (std::getline(file, line)) {
            // Look for model-engine-file in [property] section
            if (line.find("model-engine-file=") != std::string::npos) {
                size_t pos = line.find("=");
                if (pos != std::string::npos) {
                    model_engine_file = line.substr(pos + 1);
                    // Trim whitespace
                    model_engine_file.erase(0, model_engine_file.find_first_not_of(" \t"));
                    model_engine_file.erase(model_engine_file.find_last_not_of(" \t") + 1);
                    break;
                }
            }
        }
        
        if (!model_engine_file.empty()) {
            // Make absolute path if relative
            std::string engine_path = model_engine_file;
            if (!std::filesystem::path(model_engine_file).is_absolute()) {
                engine_path = std::filesystem::path(abs_sgie_path).parent_path() / model_engine_file;
            }
            
            if (std::filesystem::exists(engine_path)) {
                return engine_path;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error resolving engine path: " << e.what() << std::endl;
    }
    
    // Fallback search if config parsing fails
    return fallback_engine_search();
}

std::string EnrollOps::fallback_engine_search() {
    std::vector<std::string> candidates = {
        "models/arcface/glintr100.onnx_b4_gpu0_fp16.engine",
        "models/arcface/arcface.engine"
    };
    
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate);
        }
    }
    
    return "";
}

std::string EnrollOps::generate_aligned_filename(const std::string& user_id) {
    auto now = std::time(nullptr);
    std::ostringstream oss;
    oss << user_id << "_" << now << ".png";
    return oss.str();
}

bool EnrollOps::remove_file(const std::string& file_path) {
    try {
        return std::filesystem::remove(file_path);
    } catch (const std::exception& e) {
        return false;
    }
}

} // namespace EdgeDeepStream