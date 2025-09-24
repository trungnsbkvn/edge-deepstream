#include "env_utils.h"
#include <cstdlib>
#include <algorithm>
#include <cctype>

namespace EdgeDeepStream {
namespace EnvUtils {

std::string env_str(const std::string& name, const std::string& default_value) {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : default_value;
}

std::optional<bool> env_bool(const std::string& name, std::optional<bool> default_value) {
    const char* value = std::getenv(name.c_str());
    if (!value) {
        return default_value;
    }
    
    std::string str_value = value;
    std::transform(str_value.begin(), str_value.end(), str_value.begin(), ::tolower);
    
    // Remove leading/trailing whitespace
    str_value.erase(str_value.begin(), std::find_if(str_value.begin(), str_value.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    str_value.erase(std::find_if(str_value.rbegin(), str_value.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), str_value.end());
    
    if (str_value == "1" || str_value == "true" || str_value == "yes" || 
        str_value == "on" || str_value == "y" || str_value == "t") {
        return true;
    }
    if (str_value == "0" || str_value == "false" || str_value == "no" || 
        str_value == "off" || str_value == "n" || str_value == "f") {
        return false;
    }
    
    return default_value;
}

bool get_bool(const std::string& name, bool default_value) {
    auto result = env_bool(name, default_value);
    return result.has_value() ? result.value() : default_value;
}

int env_int(const std::string& name, int default_value) {
    const char* value = std::getenv(name.c_str());
    if (!value) {
        return default_value;
    }
    
    try {
        std::string str_value = value;
        // Remove leading/trailing whitespace
        str_value.erase(str_value.begin(), std::find_if(str_value.begin(), str_value.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        str_value.erase(std::find_if(str_value.rbegin(), str_value.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), str_value.end());
        
        return std::stoi(str_value);
    } catch (const std::exception&) {
        return default_value;
    }
}

double env_float(const std::string& name, double default_value) {
    const char* value = std::getenv(name.c_str());
    if (!value) {
        return default_value;
    }
    
    try {
        std::string str_value = value;
        // Remove leading/trailing whitespace
        str_value.erase(str_value.begin(), std::find_if(str_value.begin(), str_value.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        str_value.erase(std::find_if(str_value.rbegin(), str_value.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), str_value.end());
        
        return std::stod(str_value);
    } catch (const std::exception&) {
        return default_value;
    }
}

} // namespace EnvUtils
} // namespace EdgeDeepStream