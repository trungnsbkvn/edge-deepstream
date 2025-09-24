#include "mqtt_listener.h"
#include "placeholders.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>

#ifdef HAVE_PAHO_MQTT
#include <MQTTAsync.h>
#include <MQTTClientPersistence.h>

// MQTT response codes
#define MQTTASYNC_SUCCESS 0
#define MQTTASYNC_FAILURE -1

#endif

namespace EdgeDeepStream {

#ifdef HAVE_PAHO_MQTT

struct MQTTListener::Impl {
    MQTTAsync client;
    std::string server_uri;
    bool client_created;
    
    Impl() : client(nullptr), client_created(false) {}
    
    ~Impl() {
        if (client_created && client) {
            MQTTAsync_destroy(&client);
            client_created = false;
        }
    }
};

#else

// Placeholder implementation when MQTT not available
struct MQTTListener::Impl {
    std::string placeholder_server;
};

#endif

MQTTListener::MQTTListener() 
    : impl_(std::make_unique<Impl>())
    , port_(1883)
    , initialized_(false)
    , connected_(false)
    , shutdown_requested_(false) {
}

MQTTListener::~MQTTListener() {
    shutdown();
}

bool MQTTListener::initialize(const std::string& host, int port, 
                             const std::string& client_id,
                             const std::string& username,
                             const std::string& password) {
#ifdef HAVE_PAHO_MQTT
    if (initialized_.load()) {
        set_error("Already initialized");
        return false;
    }
    
    host_ = host;
    port_ = port;
    client_id_ = client_id.empty() ? "EdgeDeepStream_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()) : client_id;
    username_ = username;
    password_ = password;
    
    // Create server URI
    std::stringstream ss;
    ss << "tcp://" << host_ << ":" << port_;
    impl_->server_uri = ss.str();
    
    // Create MQTT client
    int rc = MQTTAsync_create(&impl_->client, impl_->server_uri.c_str(), 
                             client_id_.c_str(), MQTTCLIENT_PERSISTENCE_NONE, nullptr);
    
    if (rc != MQTTASYNC_SUCCESS) {
        std::stringstream error;
        error << "Failed to create MQTT client, return code: " << rc;
        set_error(error.str());
        return false;
    }
    
    impl_->client_created = true;
    
    // Set callbacks
    MQTTAsync_setCallbacks(impl_->client, this, nullptr, 
                          on_message_arrived, on_delivery_complete);
    
    initialized_ = true;
    std::cout << "MQTT client initialized: " << impl_->server_uri 
              << " (client_id: " << client_id_ << ")" << std::endl;
    
    return true;
#else
    // Placeholder implementation
    host_ = host;
    port_ = port;
    client_id_ = client_id.empty() ? "EdgeDeepStream_Placeholder" : client_id;
    username_ = username;
    password_ = password;
    
    impl_->placeholder_server = host + ":" + std::to_string(port);
    initialized_ = true;
    
    std::cout << "MQTT client initialized (placeholder mode): " 
              << host << ":" << port << std::endl;
    return true;
#endif
}

void MQTTListener::set_message_handler(MessageHandler handler) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    message_handler_ = handler;
}

void MQTTListener::set_connection_callback(ConnectionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    connection_callback_ = callback;
}

bool MQTTListener::connect() {
#ifdef HAVE_PAHO_MQTT
    if (!initialized_.load()) {
        set_error("MQTT client not initialized");
        return false;
    }
    
    if (connected_.load()) {
        return true; // Already connected
    }
    
    MQTTAsync_connectOptions conn_opts = MQTTAsync_connectOptions_initializer;
    conn_opts.keepAliveInterval = 30;
    conn_opts.cleansession = 1;
    conn_opts.onSuccess = on_connect_success;
    conn_opts.onFailure = on_connect_failure;
    conn_opts.context = this;
    conn_opts.automaticReconnect = 1;  // Enable automatic reconnection
    conn_opts.minRetryInterval = 1;    // Min 1 second between retries
    conn_opts.maxRetryInterval = 60;   // Max 60 seconds between retries
    
    // Set username/password if provided
    if (!username_.empty()) {
        conn_opts.username = username_.c_str();
        if (!password_.empty()) {
            conn_opts.password = password_.c_str();
        }
    }
    
    int rc = MQTTAsync_connect(impl_->client, &conn_opts);
    if (rc != MQTTASYNC_SUCCESS) {
        std::stringstream error;
        error << "Failed to start connection, return code: " << rc;
        set_error(error.str());
        return false;
    }
    
    std::cout << "MQTT connection initiated to " << impl_->server_uri << std::endl;
    return true;
#else
    // Placeholder implementation
    connected_ = true;
    std::cout << "MQTT connected (placeholder mode): " << impl_->placeholder_server << std::endl;
    handle_connection_event(true, "Placeholder connection successful");
    return true;
#endif
}

void MQTTListener::disconnect() {
#ifdef HAVE_PAHO_MQTT
    if (connected_.load() && impl_->client) {
        MQTTAsync_disconnectOptions disc_opts = MQTTAsync_disconnectOptions_initializer;
        disc_opts.timeout = 3000; // 3 second timeout
        
        int rc = MQTTAsync_disconnect(impl_->client, &disc_opts);
        if (rc == MQTTASYNC_SUCCESS) {
            connected_ = false;
            std::cout << "MQTT disconnection initiated" << std::endl;
        }
    }
#else
    connected_ = false;
    std::cout << "MQTT disconnected (placeholder mode)" << std::endl;
    handle_connection_event(false, "Placeholder disconnection");
#endif
}

bool MQTTListener::is_connected() const {
    return connected_.load();
}

bool MQTTListener::subscribe(const std::string& topic, int qos) {
#ifdef HAVE_PAHO_MQTT
    if (!connected_.load()) {
        set_error("Not connected to MQTT broker");
        return false;
    }
    
    MQTTAsync_responseOptions opts = MQTTAsync_responseOptions_initializer;
    opts.context = this;
    
    int rc = MQTTAsync_subscribe(impl_->client, topic.c_str(), qos, &opts);
    if (rc == MQTTASYNC_SUCCESS) {
        std::cout << "MQTT subscribed to topic: " << topic << " (QoS " << qos << ")" << std::endl;
        return true;
    } else {
        std::stringstream error;
        error << "Failed to subscribe to topic " << topic << ", return code: " << rc;
        set_error(error.str());
        return false;
    }
#else
    std::cout << "MQTT subscribed (placeholder): " << topic << " (QoS " << qos << ")" << std::endl;
    return true;
#endif
}

bool MQTTListener::unsubscribe(const std::string& topic) {
#ifdef HAVE_PAHO_MQTT
    if (!connected_.load()) {
        set_error("Not connected to MQTT broker");
        return false;
    }
    
    MQTTAsync_responseOptions opts = MQTTAsync_responseOptions_initializer;
    opts.context = this;
    
    int rc = MQTTAsync_unsubscribe(impl_->client, topic.c_str(), &opts);
    if (rc == MQTTASYNC_SUCCESS) {
        std::cout << "MQTT unsubscribed from topic: " << topic << std::endl;
        return true;
    } else {
        std::stringstream error;
        error << "Failed to unsubscribe from topic " << topic << ", return code: " << rc;
        set_error(error.str());
        return false;
    }
#else
    std::cout << "MQTT unsubscribed (placeholder): " << topic << std::endl;
    return true;
#endif
}

bool MQTTListener::publish(const std::string& topic, const std::string& payload, 
                          int qos, bool retain) {
#ifdef HAVE_PAHO_MQTT
    if (!connected_.load()) {
        set_error("Not connected to MQTT broker");
        return false;
    }
    
    MQTTAsync_message pubmsg = MQTTAsync_message_initializer;
    MQTTAsync_responseOptions opts = MQTTAsync_responseOptions_initializer;
    
    pubmsg.payload = (void*)payload.c_str();
    pubmsg.payloadlen = payload.length();
    pubmsg.qos = qos;
    pubmsg.retained = retain ? 1 : 0;
    opts.context = this;
    
    int rc = MQTTAsync_sendMessage(impl_->client, topic.c_str(), &pubmsg, &opts);
    if (rc == MQTTASYNC_SUCCESS) {
        return true;
    } else {
        std::stringstream error;
        error << "Failed to publish message to topic " << topic << ", return code: " << rc;
        set_error(error.str());
        return false;
    }
#else
    std::cout << "MQTT publish (placeholder): " << topic 
              << " -> " << payload.substr(0, 100) 
              << (payload.length() > 100 ? "..." : "") << std::endl;
    return true;
#endif
}

bool MQTTListener::publish_response(int cmd, const std::string& cam_id, 
                                   const std::string& user_id, const std::string& cmd_id, 
                                   int status) {
    try {
        // Format core_v2 response: "3;cmd;cam_or_zero;user_id_or_cam;cmd_id;status"
        std::stringstream payload;
        payload << "3;"                                           // RET_RESP
                << cmd << ";"                                     // command
                << (cam_id.empty() ? "0" : cam_id) << ";"        // cam id or 0
                << (user_id.empty() ? cam_id : user_id) << ";"   // user id or cam id
                << (cmd_id.empty() ? "" : cmd_id) << ";"         // command id token
                << status;                                        // status code
        
        // Use default response topic (can be configured later)
        std::string response_topic = "/local/core/v2/ai/response";
        return publish(response_topic, payload.str(), 2, false);
    } catch (const std::exception& e) {
        set_error("Failed to format response: " + std::string(e.what()));
        return false;
    }
}

void MQTTListener::shutdown() {
    if (shutdown_requested_.exchange(true)) {
        return; // Already shutting down
    }
    
    disconnect();
    initialized_ = false;
    connected_ = false;
    
    std::cout << "MQTT client shutdown complete" << std::endl;
}

std::string MQTTListener::get_last_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

// Private methods

void MQTTListener::set_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = error;
    std::cerr << "MQTT Error: " << error << std::endl;
}

void MQTTListener::handle_connection_event(bool connected, const std::string& reason) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (connection_callback_) {
        try {
            connection_callback_(connected, reason);
        } catch (const std::exception& e) {
            std::cerr << "Exception in connection callback: " << e.what() << std::endl;
        }
    }
}

void MQTTListener::handle_message_event(const std::string& topic, const std::string& payload) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (message_handler_) {
        try {
            message_handler_(topic, payload);
        } catch (const std::exception& e) {
            std::cerr << "Exception in message handler: " << e.what() << std::endl;
        }
    }
}

// Static callback implementations

#ifdef HAVE_PAHO_MQTT

void MQTTListener::on_connect_success(void* context, MQTTAsync_successData* response) {
    auto* listener = static_cast<MQTTListener*>(context);
    if (listener) {
        listener->connected_ = true;
        std::cout << "MQTT connection successful" << std::endl;
        listener->handle_connection_event(true, "Connection successful");
    }
}

void MQTTListener::on_connect_failure(void* context, MQTTAsync_failureData* response) {
    auto* listener = static_cast<MQTTListener*>(context);
    if (listener) {
        listener->connected_ = false;
        std::string reason = "Connection failed";
        if (response && response->message) {
            reason += ": " + std::string(response->message);
        }
        if (response) {
            reason += " (code: " + std::to_string(response->code) + ")";
        }
        listener->set_error(reason);
        listener->handle_connection_event(false, reason);
    }
}

void MQTTListener::on_disconnect(void* context, MQTTAsync_successData* response) {
    auto* listener = static_cast<MQTTListener*>(context);
    if (listener) {
        listener->connected_ = false;
        std::cout << "MQTT disconnected" << std::endl;
        listener->handle_connection_event(false, "Disconnected");
    }
}

int MQTTListener::on_message_arrived(void* context, char* topic_name, int topic_len, MQTTAsync_message* message) {
    auto* listener = static_cast<MQTTListener*>(context);
    if (!listener) {
        return 1; // Message processed
    }
    
    try {
        std::string topic = topic_name ? std::string(topic_name) : "";
        std::string payload;
        
        if (message && message->payload) {
            payload = std::string(static_cast<char*>(message->payload), message->payloadlen);
        }
        
        listener->handle_message_event(topic, payload);
        
        // Free the message
        MQTTAsync_freeMessage(&message);
        MQTTAsync_free(topic_name);
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in message processing: " << e.what() << std::endl;
        // Still free the message
        if (message) MQTTAsync_freeMessage(&message);
        if (topic_name) MQTTAsync_free(topic_name);
    }
    
    return 1; // Message processed successfully
}

void MQTTListener::on_delivery_complete(void* context, MQTTAsync_token token) {
    // Message delivery confirmed - could be used for QoS tracking if needed
    (void)context;
    (void)token;
}

#else

// Placeholder callback implementations (empty)
void MQTTListener::on_connect_success(void* context, void* response) {}
void MQTTListener::on_connect_failure(void* context, void* response) {}
void MQTTListener::on_disconnect(void* context, void* response) {}
int MQTTListener::on_message_arrived(void* context, char* topic_name, int topic_len, void* message) { return 1; }
void MQTTListener::on_delivery_complete(void* context, int token) {}

#endif

// EventSender implementation (placeholder for now)
EventSender::EventSender() : initialized_(false) {}
EventSender::~EventSender() = default;

bool EventSender::initialize(const std::string& endpoint) {
    endpoint_ = endpoint;
    initialized_ = true;
    std::cout << "EventSender initialized (placeholder): " << endpoint << std::endl;
    return true;
}

bool EventSender::send_event(const std::string& event_type, const std::string& data) {
    if (!initialized_) return false;
    std::cout << "Sending event (placeholder): " << event_type << " -> " << data << std::endl;
    return true;
}

void EventSender::shutdown() {
    initialized_ = false;
    std::cout << "EventSender shutdown (placeholder)" << std::endl;
}

} // namespace EdgeDeepStream