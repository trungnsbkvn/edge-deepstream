#include "event_sender.h"
#include "env_utils.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <arpa/inet.h>

namespace EdgeDeepStream {

EventSender::EventSender() 
    : socket_fd_(-1)
    , debug_enabled_(false)
{
    debug_enabled_ = EnvUtils::get_bool("EVENT_SENDER_DEBUG", false);
}

EventSender::EventSender(const std::string& socket_path)
    : socket_path_(socket_path)
    , socket_fd_(-1)
    , debug_enabled_(false)
{
    debug_enabled_ = EnvUtils::get_bool("EVENT_SENDER_DEBUG", false);
}

EventSender::~EventSender() {
    close();
}

EventSender::EventSender(EventSender&& other) noexcept
    : socket_path_(std::move(other.socket_path_))
    , socket_fd_(other.socket_fd_)
    , debug_enabled_(other.debug_enabled_)
{
    other.socket_fd_ = -1;
}

EventSender& EventSender::operator=(EventSender&& other) noexcept {
    if (this != &other) {
        close();
        socket_path_ = std::move(other.socket_path_);
        socket_fd_ = other.socket_fd_;
        debug_enabled_ = other.debug_enabled_;
        other.socket_fd_ = -1;
    }
    return *this;
}

bool EventSender::initialize(const std::string& socket_path) {
    close();  // Close any existing connection
    socket_path_ = socket_path;
    return true;  // Lazy connection on first send
}

bool EventSender::send(const std::string& event_text, const std::vector<uint8_t>& img_data) {
    return send(event_text, img_data.empty() ? nullptr : img_data.data(), 
                static_cast<uint32_t>(img_data.size()));
}

bool EventSender::send(const std::string& event_text) {
    return send(event_text, nullptr, 0);
}

bool EventSender::send(const std::string& event_text, const uint8_t* img_data, uint32_t img_size) {
    if (socket_path_.empty()) {
        if (debug_enabled_) {
            std::cerr << "[EVENT_SENDER] No socket path configured" << std::endl;
        }
        return false;
    }
    
    if (!ensure_connection()) {
        return false;
    }
    
    try {
        auto packet = create_packet(event_text, img_data, img_size);
        bool success = send_buffer(packet);
        
        if (debug_enabled_ && success) {
            std::cout << "[EVENT_SENDER] sent packet bytes=" << packet.size()
                      << " text_len=" << event_text.length()
                      << " img_len=" << img_size << std::endl;
        }
        
        return success;
    } catch (const std::exception& e) {
        if (debug_enabled_) {
            std::cerr << "[EVENT_SENDER] send error: " << e.what() << std::endl;
        }
        close();  // Reset connection on failure
        return false;
    }
}

bool EventSender::is_connected() const {
    return socket_fd_ >= 0;
}

void EventSender::close() {
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

const std::string& EventSender::get_socket_path() const {
    return socket_path_;
}

bool EventSender::ensure_connection() {
    if (socket_fd_ >= 0) {
        return true;  // Already connected
    }
    
    socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        if (debug_enabled_) {
            std::cerr << "[EVENT_SENDER] Failed to create socket: " << strerror(errno) << std::endl;
        }
        return false;
    }
    
    // Set socket timeout
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;  // 0.2 seconds
    setsockopt(socket_fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
    
    if (connect(socket_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (debug_enabled_) {
            std::cerr << "[EVENT_SENDER] Failed to connect to " << socket_path_ 
                      << ": " << strerror(errno) << std::endl;
        }
        ::close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    return true;
}

bool EventSender::send_buffer(const std::vector<uint8_t>& buffer) {
    if (socket_fd_ < 0) {
        return false;
    }
    
    size_t total_sent = 0;
    size_t remaining = buffer.size();
    
    while (remaining > 0) {
        ssize_t sent = ::send(socket_fd_, buffer.data() + total_sent, remaining, MSG_NOSIGNAL);
        if (sent <= 0) {
            if (debug_enabled_) {
                std::cerr << "[EVENT_SENDER] send() failed: " << strerror(errno) << std::endl;
            }
            return false;
        }
        total_sent += sent;
        remaining -= sent;
    }
    
    return true;
}

std::vector<uint8_t> EventSender::create_packet(const std::string& event_text, 
                                               const uint8_t* img_data, 
                                               uint32_t img_size) {
    // Convert string to UTF-8 bytes
    std::vector<uint8_t> text_bytes(event_text.begin(), event_text.end());
    uint32_t str_len = static_cast<uint32_t>(text_bytes.size());
    
    // Calculate total payload length (excluding the totalLen field itself)
    uint32_t total_len = 4 + str_len + 4 + img_size;  // strLen + str + imgSize + img
    
    // Create packet buffer
    std::vector<uint8_t> buffer;
    buffer.reserve(4 + total_len);
    
    // Write totalLen (big endian)
    uint32_t total_len_be = htonl(total_len);
    buffer.insert(buffer.end(), 
                  reinterpret_cast<uint8_t*>(&total_len_be),
                  reinterpret_cast<uint8_t*>(&total_len_be) + 4);
    
    // Write strLen (big endian)
    uint32_t str_len_be = htonl(str_len);
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&str_len_be),
                  reinterpret_cast<uint8_t*>(&str_len_be) + 4);
    
    // Write string data
    if (str_len > 0) {
        buffer.insert(buffer.end(), text_bytes.begin(), text_bytes.end());
    }
    
    // Write imgSize (big endian)
    uint32_t img_size_be = htonl(img_size);
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&img_size_be),
                  reinterpret_cast<uint8_t*>(&img_size_be) + 4);
    
    // Write image data
    if (img_size > 0 && img_data != nullptr) {
        buffer.insert(buffer.end(), img_data, img_data + img_size);
    }
    
    return buffer;
}

} // namespace EdgeDeepStream