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

/**
 * MQTT client for receiving enrollment commands and sending notifications
 */
class MQTTListener {
public:
    MQTTListener();
    ~MQTTListener();
    
    // Disable copy, enable move
    MQTTListener(const MQTTListener&) = delete;
    MQTTListener& operator=(const MQTTListener&) = delete;
    MQTTListener(MQTTListener&&) = default;
    MQTTListener& operator=(MQTTListener&&) = default;
    
    bool initialize(const std::string& broker, int port);
    void subscribe(const std::string& topic);
    void publish(const std::string& topic, const std::string& message);
    void shutdown();
    
private:
    bool initialized_;
    std::string broker_;
    int port_;
};

// TensorRT implementation moved to tensorrt_infer.h/cpp

} // namespace EdgeDeepStream