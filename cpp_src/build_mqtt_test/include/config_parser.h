#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <gst/gst.h>

namespace EdgeDeepStream {

class Config;

class ConfigParser {
public:
    static std::unique_ptr<Config> parse_toml(const std::string& file_path);
    static std::unique_ptr<Config> parse_config_file(const std::string& file_path);
    
    // Helper functions for setting GStreamer properties
    static void set_element_properties(GstElement* element, 
                                     const std::map<std::string, std::string>& properties);
    static void set_tracker_properties(GstElement* tracker, const std::string& config_path);
    
    // Face loading
    static std::map<std::string, std::vector<float>> load_faces(const std::string& path);
    
private:
    static std::string trim(const std::string& str);
    static std::pair<std::string, std::string> split_key_value(const std::string& line);
};

} // namespace EdgeDeepStream