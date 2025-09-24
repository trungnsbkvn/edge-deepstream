#include "status_codes.h"

namespace EdgeDeepStream {
namespace StatusCodes {

const std::map<int, std::string> STATUS_DESCRIPTIONS = {
    {STATUS_OK_GENERIC, "ok"},
    {STATUS_ENROLL_SUCCESS, "enroll_success"},
    {STATUS_DELETE_SUCCESS, "delete_success"},
    {STATUS_ALREADY_EXISTS, "already_exists"},
    {STATUS_DUPLICATE_OTHER, "duplicate_other_user"},
    {STATUS_INTRA_USER_MISMATCH, "intra_user_mismatch"},
    {STATUS_INVALID_REQUEST, "invalid_request"},
    {STATUS_ALIGN_FAIL, "align_fail"},
    {STATUS_EMBED_FAIL, "embed_fail"},
    {STATUS_LOW_QUALITY_BLUR, "low_quality_blur"},
    {STATUS_UNKNOWN_ERROR, "unknown_error"},
};

std::string get_status_description(int status_code) {
    auto it = STATUS_DESCRIPTIONS.find(status_code);
    if (it != STATUS_DESCRIPTIONS.end()) {
        return it->second;
    }
    return "unknown_status";
}

bool is_success_status(int status_code) {
    return status_code == STATUS_OK_GENERIC || 
           status_code == STATUS_ENROLL_SUCCESS || 
           status_code == STATUS_DELETE_SUCCESS || 
           status_code == STATUS_ALREADY_EXISTS;
}

} // namespace StatusCodes
} // namespace EdgeDeepStream