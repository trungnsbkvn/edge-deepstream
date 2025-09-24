#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace EdgeDeepStream {

/**
 * Event sender for status updates and notifications
 */
class EventSender {
public:
    EventSender();
    ~EventSender();
    
    // Disable copy, enable move
    EventSender(const EventSender&) = delete;
    EventSender& operator=(const EventSender&) = delete;
    EventSender(EventSender&&) = default;
    EventSender& operator=(EventSender&&) = default;
    
    bool initialize(const std::string& endpoint);
    bool send_event(const std::string& event_type, const std::string& data);
    void shutdown();
    
private:
    bool initialized_;
    std::string endpoint_;
};

// MQTT implementation moved to mqtt_listener.h/cpp

// TensorRT implementation moved to tensorrt_infer.h/cpp

} // namespace EdgeDeepStream