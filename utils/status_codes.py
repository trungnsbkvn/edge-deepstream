# Centralized response status codes for MQTT enrollment/deletion
# Keep numeric values stable to avoid breaking external integrations.

# General success codes
STATUS_OK_GENERIC = 0            # Generic success / ack for non-enroll commands (e.g. source add/remove)
STATUS_ENROLL_SUCCESS = 2        # Enrollment success (add / update person)
STATUS_DELETE_SUCCESS = 3        # Deletion success

# Validation / logical conditions
STATUS_ALREADY_EXISTS = 2        # Reuse enroll success for idempotent same-image add
STATUS_DUPLICATE_OTHER = 6       # Duplicate face belongs to another user
STATUS_INTRA_USER_MISMATCH = 7   # New image for same user not similar enough

# Input / request issues
STATUS_INVALID_REQUEST = 1       # Missing user_id or image path
STATUS_ALIGN_FAIL = 4            # Failed to read or align image
STATUS_EMBED_FAIL = 5            # Embedding / engine failure
STATUS_LOW_QUALITY_BLUR = 10     # Image too blurry / below Laplacian variance threshold

# Generic failure
STATUS_UNKNOWN_ERROR = 9         # Catch-all unexpected error

# Map for reverse lookup / debugging
STATUS_DESCRIPTIONS = {
    STATUS_OK_GENERIC: "ok",
    STATUS_ENROLL_SUCCESS: "enroll_success",
    STATUS_DELETE_SUCCESS: "delete_success",
    STATUS_ALREADY_EXISTS: "already_exists",
    STATUS_DUPLICATE_OTHER: "duplicate_other_user",
    STATUS_INTRA_USER_MISMATCH: "intra_user_mismatch",
    STATUS_INVALID_REQUEST: "invalid_request",
    STATUS_ALIGN_FAIL: "align_fail",
    STATUS_EMBED_FAIL: "embed_fail",
    STATUS_LOW_QUALITY_BLUR: "low_quality_blur",
    STATUS_UNKNOWN_ERROR: "unknown_error",
}

__all__ = [
    'STATUS_OK_GENERIC',
    'STATUS_ENROLL_SUCCESS',
    'STATUS_DELETE_SUCCESS',
    'STATUS_ALREADY_EXISTS',
    'STATUS_DUPLICATE_OTHER',
    'STATUS_INTRA_USER_MISMATCH',
    'STATUS_INVALID_REQUEST',
    'STATUS_ALIGN_FAIL',
    'STATUS_EMBED_FAIL',
    'STATUS_LOW_QUALITY_BLUR',
    'STATUS_UNKNOWN_ERROR',
    'STATUS_DESCRIPTIONS'
]
