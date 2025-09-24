#include "config_parser.h"
#include "edge_deepstream.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <iostream>

#ifdef HAVE_CPPTOML
#include "cpptoml.h"
#endif

namespace EdgeDeepStream {

std::unique_ptr<Config> ConfigParser::parse_toml(const std::string& file_path) {
#ifdef HAVE_CPPTOML
    try {
        auto config = std::make_unique<Config>();
        auto toml_config = cpptoml::parse_file(file_path);
        
        // Convert cpptoml structure to our Config format
        for (const auto& table_pair : *toml_config) {
            const std::string& section_name = table_pair.first;
            auto table = table_pair.second->as_table();
            
            if (table) {
                std::map<std::string, std::string> section_map;
                
                for (const auto& entry : *table) {
                    const std::string& key = entry.first;
                    auto value = entry.second;
                    
                    if (auto str_val = value->as<std::string>()) {
                        section_map[key] = str_val->get();
                    } else if (auto int_val = value->as<int64_t>()) {
                        section_map[key] = std::to_string(int_val->get());
                    } else if (auto bool_val = value->as<bool>()) {
                        section_map[key] = bool_val->get() ? "1" : "0";
                    } else if (auto double_val = value->as<double>()) {
                        section_map[key] = std::to_string(double_val->get());
                    } else {
                        // Try to convert to string representation
                        std::stringstream ss;
                        ss << *value;
                        section_map[key] = ss.str();
                    }
                }
                
                config->sections[section_name] = std::move(section_map);
            }
        }
        
        return config;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing TOML file " << file_path << ": " << e.what() << std::endl;
        return nullptr;
    }
#else
    // Fallback: simple TOML-like parser
    return parse_config_file(file_path);
#endif
}

std::unique_ptr<Config> ConfigParser::parse_config_file(const std::string& file_path) {
    auto config = std::make_unique<Config>();
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open config file: " << file_path << std::endl;
        return nullptr;
    }
    
    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Section header
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            current_section = trim(current_section);
            config->sections[current_section] = std::map<std::string, std::string>();
            continue;
        }
        
        // Key-value pair
        if (!current_section.empty()) {
            auto [key, value] = split_key_value(line);
            if (!key.empty()) {
                // Remove quotes from string values
                if (value.size() >= 2 && 
                    ((value.front() == '"' && value.back() == '"') ||
                     (value.front() == '\'' && value.back() == '\''))) {
                    value = value.substr(1, value.length() - 2);
                }
                config->sections[current_section][key] = value;
            }
        }
    }
    
    return config;
}

void ConfigParser::set_element_properties(GstElement* element, 
                                        const std::map<std::string, std::string>& properties) {
    if (!element) return;
    
    for (const auto& [key, value] : properties) {
        // Convert string value to appropriate type based on property
        GParamSpec* spec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), key.c_str());
        if (!spec) {
            std::cout << "Warning: Property '" << key << "' not found on element" << std::endl;
            continue;
        }
        
        std::cout << "Setting property " << key << " = " << value << std::endl;
        
        GType type = G_PARAM_SPEC_VALUE_TYPE(spec);
        
        if (type == G_TYPE_STRING) {
            g_object_set(element, key.c_str(), value.c_str(), NULL);
        } else if (type == G_TYPE_INT) {
            try {
                int int_val = std::stoi(value);
                g_object_set(element, key.c_str(), int_val, NULL);
            } catch (...) {
                std::cerr << "Error: Cannot convert '" << value << "' to integer for property '" << key << "'" << std::endl;
            }
        } else if (type == G_TYPE_UINT) {
            try {
                unsigned int uint_val = static_cast<unsigned int>(std::stoul(value));
                g_object_set(element, key.c_str(), uint_val, NULL);
            } catch (...) {
                std::cerr << "Error: Cannot convert '" << value << "' to unsigned integer for property '" << key << "'" << std::endl;
            }
        } else if (type == G_TYPE_BOOLEAN) {
            std::string lower_value = value;
            std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
            bool bool_val = (lower_value == "1" || lower_value == "true" || 
                           lower_value == "yes" || lower_value == "on");
            g_object_set(element, key.c_str(), bool_val, NULL);
        } else if (type == G_TYPE_FLOAT) {
            try {
                float float_val = std::stof(value);
                g_object_set(element, key.c_str(), float_val, NULL);
            } catch (...) {
                std::cerr << "Error: Cannot convert '" << value << "' to float for property '" << key << "'" << std::endl;
            }
        } else if (type == G_TYPE_DOUBLE) {
            try {
                double double_val = std::stod(value);
                g_object_set(element, key.c_str(), double_val, NULL);
            } catch (...) {
                std::cerr << "Error: Cannot convert '" << value << "' to double for property '" << key << "'" << std::endl;
            }
        } else if (type == G_TYPE_UINT64) {
            try {
                uint64_t uint64_val = std::stoull(value);
                g_object_set(element, key.c_str(), uint64_val, NULL);
            } catch (...) {
                std::cerr << "Error: Cannot convert '" << value << "' to uint64 for property '" << key << "'" << std::endl;
            }
        } else {
            // Try setting as string for unknown types
            g_object_set(element, key.c_str(), value.c_str(), NULL);
        }
    }
}

void ConfigParser::set_tracker_properties(GstElement* tracker, const std::string& config_path) {
    if (!tracker || config_path.empty()) return;
    
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open tracker config file: " << config_path << std::endl;
        return;
    }
    
    std::string line;
    std::string current_section;
    std::map<std::string, std::string> tracker_properties;
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Section header
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            current_section = trim(current_section);
            continue;
        }
        
        // Only process [tracker] section
        if (current_section == "tracker") {
            auto [key, value] = split_key_value(line);
            if (!key.empty()) {
                tracker_properties[key] = value;
            }
        }
    }
    
    // Set tracker-specific properties
    for (const auto& [key, value] : tracker_properties) {
        std::cout << "Setting tracker property " << key << " = " << value << std::endl;
        
        if (key == "tracker-width") {
            try {
                int width = std::stoi(value);
                g_object_set(tracker, "tracker-width", width, NULL);
            } catch (...) {}
        } else if (key == "tracker-height") {
            try {
                int height = std::stoi(value);
                g_object_set(tracker, "tracker-height", height, NULL);
            } catch (...) {}
        } else if (key == "gpu-id") {
            try {
                int gpu_id = std::stoi(value);
                g_object_set(tracker, "gpu_id", gpu_id, NULL);
            } catch (...) {}
        } else if (key == "ll-lib-file") {
            g_object_set(tracker, "ll-lib-file", value.c_str(), NULL);
        } else if (key == "ll-config-file") {
            g_object_set(tracker, "ll-config-file", value.c_str(), NULL);
        }
    }
}

std::map<std::string, std::vector<float>> ConfigParser::load_faces(const std::string& path) {
    std::map<std::string, std::vector<float>> loaded_faces;
    
    try {
        if (!std::filesystem::exists(path)) {
            std::cout << "Face directory does not exist: " << path << std::endl;
            return loaded_faces;
        }
        
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                const std::string filename = entry.path().filename().string();
                
                // Check for .npy files (would need numpy-equivalent library)
                if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".npy") {
                    // TODO: Implement .npy file loading
                    // For now, just note that the file exists
                    std::string name = filename.substr(0, filename.length() - 4);  // Remove .npy
                    loaded_faces[name] = std::vector<float>();  // Placeholder
                    std::cout << "Found face file (placeholder): " << name << std::endl;
                }
                
                // Check for other image formats that could be processed
                if ((filename.length() > 4 && filename.substr(filename.length() - 4) == ".jpg") ||
                    (filename.length() > 5 && filename.substr(filename.length() - 5) == ".jpeg") ||
                    (filename.length() > 4 && filename.substr(filename.length() - 4) == ".png") ||
                    (filename.length() > 4 && filename.substr(filename.length() - 4) == ".bmp")) {
                    // TODO: Implement image processing and feature extraction
                    std::string name = filename.substr(0, filename.find_last_of('.'));
                    loaded_faces[name] = std::vector<float>();  // Placeholder
                    std::cout << "Found image file (placeholder): " << name << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading faces from " << path << ": " << e.what() << std::endl;
    }
    
    return loaded_faces;
}

std::string ConfigParser::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::pair<std::string, std::string> ConfigParser::split_key_value(const std::string& line) {
    size_t pos = line.find('=');
    if (pos == std::string::npos) {
        return {"", ""};
    }
    
    std::string key = trim(line.substr(0, pos));
    std::string value = trim(line.substr(pos + 1));
    
    return {key, value};
}

} // namespace EdgeDeepStream