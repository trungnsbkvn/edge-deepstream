#pragma once

#include <string>
#include <map>

namespace EdgeDeepStream {
namespace StatusCodes {

// Status codes (matching Python implementation)
constexpr int STATUS_OK_GENERIC = 0;
constexpr int STATUS_ENROLL_SUCCESS = 2;
constexpr int STATUS_DELETE_SUCCESS = 3;
constexpr int STATUS_ALREADY_EXISTS = 2;
constexpr int STATUS_DUPLICATE_OTHER = 6;
constexpr int STATUS_INTRA_USER_MISMATCH = 7;
constexpr int STATUS_INVALID_REQUEST = 1;
constexpr int STATUS_ALIGN_FAIL = 4;
constexpr int STATUS_EMBED_FAIL = 5;
constexpr int STATUS_LOW_QUALITY_BLUR = 10;
constexpr int STATUS_UNKNOWN_ERROR = 9;

// Status descriptions
extern const std::map<int, std::string> STATUS_DESCRIPTIONS;

// Helper functions
std::string get_status_description(int status_code);
bool is_success_status(int status_code);

} // namespace StatusCodes
} // namespace EdgeDeepStream