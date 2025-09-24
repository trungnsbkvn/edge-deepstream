#include "mqtt_listener.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>
#include <sstream>
#include <vector>

using namespace EdgeDeepStream;

std::atomic<bool> g_running{true};

void signal_handler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    g_running = false;
}

void parse_face_command(const std::string& payload, MQTTListener& mqtt) {
    // Parse core_v2 protocol commands
    // Format: "req_type;cmd;cam_id;rtsp;face_enable;user_id;user_name;user_img;cmd_id"
    
    std::cout << "Parsing command: " << payload << std::endl;
    
    // Simple parsing (in production, use proper delimiter parsing)
    std::vector<std::string> tokens;
    std::stringstream ss(payload);
    std::string token;
    
    while (std::getline(ss, token, ';')) {
        tokens.push_back(token);
    }
    
    if (tokens.size() < 2) {
        std::cout << "Invalid command format" << std::endl;
        return;
    }
    
    try {
        int cmd = std::stoi(tokens[1]);
        std::string cam_id = tokens.size() > 2 ? tokens[2] : "";
        std::string cmd_id = tokens.size() > 8 ? tokens[8] : "";
        
        std::cout << "Command: " << cmd << ", Camera ID: " << cam_id << std::endl;
        
        if (cmd == 25) {  // CMD_BOX_ENABLE_SERVICE
            std::string rtsp = tokens.size() > 3 ? tokens[3] : "";
            int face_enable = tokens.size() > 4 ? std::stoi(tokens[4]) : 0;
            
            if (face_enable == 1) {
                std::cout << "Enable face recognition for camera: " << cam_id << std::endl;
                std::cout << "RTSP URL: " << rtsp << std::endl;
                // Send success response
                mqtt.publish_response(25, cam_id, "", cmd_id, 0);
            } else {
                std::cout << "Disable face recognition for camera: " << cam_id << std::endl;
                mqtt.publish_response(25, cam_id, "", cmd_id, 0);
            }
        }
        else if (cmd == 26) {  // CMD_BOX_DISABLE_SERVICE
            std::cout << "Disable service for camera: " << cam_id << std::endl;
            mqtt.publish_response(26, cam_id, "", cmd_id, 0);
        }
        else if (cmd == 60) {  // CMD_BOX_PUT_PERSON
            std::string user_id = tokens.size() > 5 ? tokens[5] : "";
            std::string user_name = tokens.size() > 6 ? tokens[6] : "";
            std::string user_img = tokens.size() > 7 ? tokens[7] : "";
            
            std::cout << "Add person: " << user_id << " (" << user_name << ")" << std::endl;
            std::cout << "Image path: " << user_img << std::endl;
            
            // Simulate enrollment process
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            mqtt.publish_response(60, "", user_id, cmd_id, 2);  // Status 2 = success
        }
        else if (cmd == 61) {  // CMD_BOX_DEL_PERSON
            std::string user_id = tokens.size() > 5 ? tokens[5] : "";
            
            std::cout << "Delete person: " << user_id << std::endl;
            
            // Simulate deletion process
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            mqtt.publish_response(61, "", user_id, cmd_id, 3);  // Status 3 = deleted
        }
        else {
            std::cout << "Unknown command: " << cmd << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing command: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== MQTT Client Test ===" << std::endl;
    
    // Install signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    MQTTListener mqtt;
    
    // Set up callbacks
    mqtt.set_message_handler([&mqtt](const std::string& topic, const std::string& payload) {
        std::cout << "Received message on topic '" << topic << "':" << std::endl;
        std::cout << "  Payload: " << payload << std::endl;
        parse_face_command(payload, mqtt);
        std::cout << std::endl;
    });
    
    mqtt.set_connection_callback([](bool connected, const std::string& reason) {
        std::cout << "Connection status: " << (connected ? "CONNECTED" : "DISCONNECTED") 
                  << " - " << reason << std::endl;
    });
    
    // Initialize MQTT client
    std::string host = "localhost";
    int port = 1883;
    std::string client_id = "EdgeDeepStreamTest";
    
    std::cout << "Initializing MQTT client..." << std::endl;
    if (!mqtt.initialize(host, port, client_id)) {
        std::cerr << "Failed to initialize MQTT client: " << mqtt.get_last_error() << std::endl;
        return 1;
    }
    
    // Connect to broker
    std::cout << "Connecting to MQTT broker..." << std::endl;
    if (!mqtt.connect()) {
        std::cerr << "Failed to connect to MQTT broker: " << mqtt.get_last_error() << std::endl;
        return 1;
    }
    
    // Wait a moment for connection
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Subscribe to request topic
    std::string request_topic = "/local/core/v2/ai/request";
    std::cout << "Subscribing to topic: " << request_topic << std::endl;
    if (!mqtt.subscribe(request_topic, 2)) {
        std::cerr << "Failed to subscribe to topic: " << mqtt.get_last_error() << std::endl;
        return 1;
    }
    
    std::cout << "\nMQTT client ready! Listening for commands..." << std::endl;
    std::cout << "Send test commands to topic: " << request_topic << std::endl;
    std::cout << "\nExample commands:" << std::endl;
    std::cout << "  Enable service:  '1;25;cam001;rtsp://test:test@192.168.1.100/stream;1;;;;test123'" << std::endl;
    std::cout << "  Disable service: '1;26;cam001;;;;;;;;;test124'" << std::endl;
    std::cout << "  Add person:      '1;60;;;user123;John Doe;/path/to/image.jpg;test125'" << std::endl;
    std::cout << "  Delete person:   '1;61;;;;user123;;;;test126'" << std::endl;
    std::cout << "\nPress Ctrl+C to exit..." << std::endl;
    
    // Test publishing (optional)
    if (mqtt.is_connected()) {
        std::cout << "\nPublishing test status message..." << std::endl;
        std::string test_response = "3;0;0;;system_ready;0";  // System ready response
        if (mqtt.publish("/local/core/v2/ai/response", test_response)) {
            std::cout << "Test message published successfully" << std::endl;
        }
    }
    
    // Main loop
    while (g_running.load()) {
        if (!mqtt.is_connected()) {
            std::cout << "Connection lost, attempting to reconnect..." << std::endl;
            mqtt.connect();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\nShutting down MQTT client..." << std::endl;
    mqtt.shutdown();
    
    std::cout << "MQTT test complete." << std::endl;
    return 0;
}