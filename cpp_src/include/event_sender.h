#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace EdgeDeepStream {

/**
 * Unix domain socket sender for face recognition events.
 * 
 * Packet format (Big Endian / network order):
 * - totalLen: uint32 (length of the remaining payload, not including this field)
 * - strLen:   uint32 (length of UTF-8 string)  
 * - str:      bytes  (event text)
 * - imgSize:  uint32 (length of image bytes)
 * - imgData:  bytes  (encoded image, e.g., PNG/JPEG)
 */
class EventSender {
public:
    EventSender();
    explicit EventSender(const std::string& socket_path);
    ~EventSender();
    
    // Disable copy, enable move
    EventSender(const EventSender&) = delete;
    EventSender& operator=(const EventSender&) = delete;
    EventSender(EventSender&&) noexcept;
    EventSender& operator=(EventSender&&) noexcept;
    
    /**
     * Initialize the event sender with socket path
     */
    bool initialize(const std::string& socket_path);
    
    /**
     * Send an event with optional image data
     * @param event_text The event message text
     * @param img_data Optional image data (PNG/JPEG encoded)
     * @return true if sent successfully, false otherwise
     */
    bool send(const std::string& event_text, const std::vector<uint8_t>& img_data);
    
    /**
     * Send an event with text only
     * @param event_text The event message text
     * @return true if sent successfully, false otherwise
     */
    bool send(const std::string& event_text);
    
    /**
     * Send an event with optional image data (raw pointer version)
     * @param event_text The event message text
     * @param img_data Optional image data pointer
     * @param img_size Size of image data
     * @return true if sent successfully, false otherwise
     */
    bool send(const std::string& event_text, const uint8_t* img_data, uint32_t img_size);
    
    /**
     * Check if the sender is initialized and connected
     */
    bool is_connected() const;
    
    /**
     * Close the connection
     */
    void close();
    
    /**
     * Get the socket path
     */
    const std::string& get_socket_path() const;
    
private:
    std::string socket_path_;
    int socket_fd_;
    bool debug_enabled_;
    
    /**
     * Ensure connection is established
     */
    bool ensure_connection();
    
    /**
     * Send raw buffer data
     */
    bool send_buffer(const std::vector<uint8_t>& buffer);
    
    /**
     * Create packet from event text and image data
     */
    std::vector<uint8_t> create_packet(const std::string& event_text, 
                                      const uint8_t* img_data, 
                                      uint32_t img_size);
};

} // namespace EdgeDeepStream