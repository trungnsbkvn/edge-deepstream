#pragma once

#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>

// Forward declarations for paho-mqtt types
struct MQTTAsync_struct;
typedef struct MQTTAsync_struct* MQTTAsync;

namespace EdgeDeepStream {

/**
 * MQTT client for receiving enrollment commands and sending notifications
 * Compatible with core_v2 protocol for face recognition commands
 */
class MQTTListener {
public:
    // Message handler callback: receives parsed JSON data or raw payload
    using MessageHandler = std::function<void(const std::string& topic, const std::string& payload)>;
    
    // Connection status callback
    using ConnectionCallback = std::function<void(bool connected, const std::string& reason)>;
    
    MQTTListener();
    ~MQTTListener();
    
    // Disable copy, enable move
    MQTTListener(const MQTTListener&) = delete;
    MQTTListener& operator=(const MQTTListener&) = delete;
    MQTTListener(MQTTListener&&) = default;
    MQTTListener& operator=(MQTTListener&&) = default;
    
    // Initialize MQTT client
    bool initialize(const std::string& host, int port = 1883, 
                   const std::string& client_id = "",
                   const std::string& username = "",
                   const std::string& password = "");
    
    // Set callbacks
    void set_message_handler(MessageHandler handler);
    void set_connection_callback(ConnectionCallback callback);
    
    // Connection management
    bool connect();
    void disconnect();
    bool is_connected() const;
    
    // Subscription management
    bool subscribe(const std::string& topic, int qos = 2);
    bool unsubscribe(const std::string& topic);
    
    // Publishing
    bool publish(const std::string& topic, const std::string& payload, 
                int qos = 2, bool retain = false);
    
    // High-level face recognition protocol helpers
    bool publish_response(int cmd, const std::string& cam_id, 
                         const std::string& user_id, const std::string& cmd_id, 
                         int status);
    
    // Cleanup
    void shutdown();
    
    // Status
    std::string get_last_error() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Connection parameters
    std::string host_;
    int port_;
    std::string client_id_;
    std::string username_;
    std::string password_;
    
    // State
    std::atomic<bool> initialized_;
    std::atomic<bool> connected_;
    std::atomic<bool> shutdown_requested_;
    
    // Callbacks
    MessageHandler message_handler_;
    ConnectionCallback connection_callback_;
    std::mutex callback_mutex_;
    
    // Error tracking
    mutable std::mutex error_mutex_;
    std::string last_error_;
    
    // Internal methods
    void set_error(const std::string& error);
    void handle_connection_event(bool connected, const std::string& reason);
    void handle_message_event(const std::string& topic, const std::string& payload);
    
    // Static callbacks for paho-mqtt
    static void on_connect_success(void* context, void* response);
    static void on_connect_failure(void* context, void* response);
    static void on_disconnect(void* context, void* properties);
    static int on_message_arrived(void* context, char* topic_name, int topic_len, void* message);
    static void on_delivery_complete(void* context, int token);
};

} // namespace EdgeDeepStream