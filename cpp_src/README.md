# EdgeDeepStream C++ Application

This is the C++ conversion of the EdgeDeepStream face recognition pipeline application.

## Overview

The application has been converted from Python to C++ while maintaining the same core functionality and logic:

- **GStreamer Pipeline**: RTSP source handling, DeepStream inference, tracking, and display
- **Face Recognition**: TensorRT inference with ArcFace embeddings
- **Configuration**: TOML configuration file parsing
- **MQTT Integration**: Event messaging and enrollment commands
- **Real-time Optimization**: Motion-aware buffer management and frame dropping

## Architecture

### Core Components

- `main.cpp`: Application entry point and initialization
- `pipeline.cpp`: GStreamer pipeline management
- `source_bin.cpp`: RTSP and URI source handling with dynamic linking
- `config_parser.cpp`: Configuration file parsing (TOML support)

### Utility Modules

- `env_utils.cpp`: Environment variable helpers
- `status_codes.cpp`: Status code definitions and descriptions
- `bus_call.cpp`: GStreamer bus message handling
- Placeholder implementations for:
  - `event_sender.cpp`: Event notification system
  - `mqtt_listener.cpp`: MQTT client integration
  - `faiss_index.cpp`: Face feature indexing
  - `gen_feature.cpp`: TensorRT inference engine
  - `enroll_ops.cpp`: Face enrollment operations
  - `perf_stats.cpp`: Performance monitoring

## Building

### Prerequisites

- CMake 3.16+
- GStreamer 1.0 development packages
- DeepStream SDK
- NVIDIA CUDA Toolkit
- OpenCV
- Optional: cpptoml, nlohmann/json, paho-mqtt

### Build Instructions

```bash
# Quick build
./build_cpp.sh

# Manual build
cd cpp_src
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```bash
# Run with configuration file
./edge_deepstream_cpp config/config_pipeline.toml

# Run with duration limit (milliseconds)
./edge_deepstream_cpp config/config_pipeline.toml 60000

# Compare with Python version
python python_backup/main.py config/config_pipeline.toml
```

## Configuration

The C++ version uses the same TOML configuration files as the Python version:

- `config/config_pipeline.toml`: Main pipeline configuration
- `config/config_*.txt`: DeepStream model configurations
- Environment variables: Same as Python version (DS_*, etc.)

## Migration Notes

### Maintained Features

✅ **Core Pipeline Logic**: Identical GStreamer pipeline structure  
✅ **RTSP Handling**: Dynamic codec detection and hardware decoding  
✅ **Configuration**: Full TOML configuration compatibility  
✅ **Environment Variables**: All DS_* environment variables supported  
✅ **Real-time Optimization**: Frame dropping and buffer management  
✅ **Multi-source Support**: Dynamic source addition/removal  

### Placeholder Implementations

The following modules have placeholder implementations and need full integration:

🔄 **Face Recognition**: TensorRT inference engine  
🔄 **FAISS Integration**: Face feature indexing and search  
🔄 **MQTT Client**: Event messaging and enrollment commands  
🔄 **Event System**: Notification and status reporting  
🔄 **Performance Stats**: Real-time performance monitoring  

### Key Differences

- **Memory Management**: Automatic C++ RAII vs manual Python cleanup
- **Type Safety**: Compile-time type checking vs runtime checks
- **Performance**: Direct GStreamer C API usage vs Python bindings
- **Threading**: Native C++ threading vs GIL limitations

## Development

### Adding Functionality

1. Implement placeholder classes in their respective files
2. Add required dependencies to `CMakeLists.txt`
3. Update include paths in header files
4. Test with existing Python version for comparison

### Testing

```bash
# Test basic pipeline functionality
./edge_deepstream_cpp config/config_pipeline.toml 10000

# Compare output with Python version
python python_backup/main.py config/config_pipeline.toml

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./edge_deepstream_cpp config/config_pipeline.toml 5000
```

## Troubleshooting

### Common Issues

1. **Missing DeepStream**: Ensure DeepStream SDK is installed and paths are correct
2. **GStreamer Plugins**: Check that required plugins are available
3. **NVIDIA Drivers**: Ensure compatible NVIDIA drivers are installed
4. **Config Files**: Verify all referenced config files exist

### Debug Build

```bash
cd cpp_src/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
gdb ./edge_deepstream
```

## License

Same license as the original Python implementation.