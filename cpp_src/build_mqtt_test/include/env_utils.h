#pragma once

#include <string>
#include <optional>

namespace EdgeDeepStream {
namespace EnvUtils {

// Environment variable helpers (matching Python implementation)
std::string env_str(const std::string& name, const std::string& default_value = "");
std::optional<bool> env_bool(const std::string& name, std::optional<bool> default_value = std::nullopt);
int env_int(const std::string& name, int default_value);
double env_float(const std::string& name, double default_value);

} // namespace EnvUtils
} // namespace EdgeDeepStream