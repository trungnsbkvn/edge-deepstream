cmake_minimum_required(VERSION 3.16)
project(MQTTTest VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (needed for placeholders.h)
find_package(OpenCV REQUIRED)

# Find paho-mqtt libraries
find_library(PAHO_MQTT_CPP_LIBRARY
    NAMES paho-mqttpp3
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    DOC "Paho MQTT C++ library")

find_library(PAHO_MQTT_C_LIBRARY
    NAMES paho-mqtt3as
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    DOC "Paho MQTT C library")

find_path(PAHO_MQTT_INCLUDE_DIR
    NAMES MQTTAsync.h
    PATHS /usr/local/include /usr/include
    DOC "Paho MQTT include directory")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Check for paho-mqtt availability
if(PAHO_MQTT_CPP_LIBRARY AND PAHO_MQTT_C_LIBRARY AND PAHO_MQTT_INCLUDE_DIR)
    message(STATUS "Paho MQTT found:")
    message(STATUS "  C++ library: ${PAHO_MQTT_CPP_LIBRARY}")
    message(STATUS "  C library: ${PAHO_MQTT_C_LIBRARY}")
    message(STATUS "  Include dir: ${PAHO_MQTT_INCLUDE_DIR}")
    include_directories(${PAHO_MQTT_INCLUDE_DIR})
    add_compile_definitions(HAVE_PAHO_MQTT)
    set(MQTT_LIBS ${PAHO_MQTT_CPP_LIBRARY} ${PAHO_MQTT_C_LIBRARY})
else()
    message(STATUS "Paho MQTT not found - using placeholder implementation")
    set(MQTT_LIBS "")
endif()

# Source files
set(SOURCES
    test_mqtt.cpp
    utils/mqtt_listener.cpp
)

# Create test executable
add_executable(test_mqtt ${SOURCES})

# Link libraries
target_link_libraries(test_mqtt
    ${MQTT_LIBS}
    ${OpenCV_LIBS}
    pthread
)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(test_mqtt PRIVATE -Wall -Wextra -O2)
endif()

message(STATUS "MQTT test configuration complete")
if(MQTT_LIBS)
    message(STATUS "  Real MQTT implementation will be used")
else()
    message(STATUS "  Placeholder MQTT implementation will be used")
endif()